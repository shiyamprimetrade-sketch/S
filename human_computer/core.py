"""Local-only foundation for a future human-like personal computer.

Separates goals, observed machine state, operations, evidence, and teacher feedback.
The implementation is not a replacement kernel and has no general intelligence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import platform
import shutil
import sqlite3
import threading
from typing import Any

from . import symbolic_vm as vm


def utcnow():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Result:
    kind: str
    status: str
    text: str
    evidence: dict[str, Any]
    verified: bool = False

    def as_json(self):
        return asdict(self)


class HumanComputer:
    """One user-facing goal interface with explicit, auditable internal steps."""

    def __init__(self, home: str | Path):
        self.home = Path(home).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace = self.home / 'workspace'
        self.workspace.mkdir(exist_ok=True)
        self.db_path = self.home / 'axiom_pc.sqlite3'
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        with self.connection:
            self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                at TEXT NOT NULL, kind TEXT NOT NULL, input TEXT NOT NULL,
                status TEXT NOT NULL, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memories(
                name TEXT PRIMARY KEY, value TEXT NOT NULL,
                provenance TEXT NOT NULL, updated TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS goals(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL, state TEXT NOT NULL, created TEXT NOT NULL);
            ''')
        self.memory_path = self.home / 'learned_skills.json'

    def close(self):
        self.connection.close()

    def _event(self, kind, inp, result: Result):
        with self._lock, self.connection:
            self.connection.execute(
                'INSERT INTO events(at,kind,input,status,evidence) VALUES(?,?,?,?,?)',
                (utcnow(), kind, inp[:10000], result.status, json.dumps(result.as_json(), ensure_ascii=False)))
        return result

    def memory(self):
        with self._lock:
            return vm.read_memory(self.memory_path)

    def _save_skills(self, record):
        vm.save_memory(self.memory_path, record)

    def teach(self, skill, example_in, expected_out):
        with self._lock:
            mem = self.memory()
            out = vm.teach(mem, skill, example_in, expected_out)
            if out['status'] != 'CONFLICT':
                self._save_skills(mem)
        return self._event('teach', f'{skill}: {example_in} → {expected_out}',
                           Result('learning', out['status'], json.dumps(out),
                                  {'examples': out.get('examples'), 'skill': skill}, out['status'] != 'CONFLICT'))

    def invent(self, skill, depth=4):
        with self._lock:
            mem = self.memory()
            out = vm.invent(mem, skill, depth)
            if out['status'] == 'CANDIDATE_FOUND':
                self._save_skills(mem)
        return self._event('invention', skill,
                           Result('invention', out['status'], json.dumps(out),
                                  {'skill': skill, 'candidate': out.get('program'),
                                   'searched_states': out.get('searched_states')}, False))

    def perform(self, skill, value):
        with self._lock:
            result = vm.run(self.memory(), skill, value, show_trace=True)
        return self._event('action', f'{skill}({value})', Result('action', result['status'],
                           str(result.get('output', result.get('next', ''))), result,
                           False))

    def verify(self, skill, inp, expected):
        with self._lock:
            mem = self.memory()
            out = vm.prove(mem, skill, inp, expected)
            if out['status'] in ('TEST_PASS', 'TEST_FAIL'):
                self._save_skills(mem)
        return self._event('verification', f'{skill}({inp}) == {expected}',
                           Result('verification', out['status'], json.dumps(out), out,
                                  out['status'] == 'TEST_PASS'))

    def remember(self, name, value, provenance='user'):
        if not name.strip() or len(name) > 120 or len(value) > 20000:
            raise ValueError('Invalid memory name or content')
        with self._lock, self.connection:
            self.connection.execute('INSERT OR REPLACE INTO memories(name,value,provenance,updated) VALUES(?,?,?,?)',
                                    (name, value, provenance, utcnow()))
        return self._event('memory', name, Result('memory', 'STORED', f'Remembered: {name}',
                                                  {'provenance': provenance}, True))

    def recall(self, query=''):
        with self._lock:
            rows = self.connection.execute('SELECT name,value,provenance,updated FROM memories WHERE name LIKE ? OR value LIKE ? ORDER BY updated DESC LIMIT 30',
                ('%' + query + '%', '%' + query + '%')).fetchall()
        return [dict(x) for x in rows]

    def status(self):
        usage = shutil.disk_usage(self.home)
        return {'platform': platform.platform(), 'processor': platform.machine(),
                'python': platform.python_version(),
                'workspace': str(self.workspace), 'disk_free_bytes': usage.free,
                'disk_total_bytes': usage.total,
                'learned_skills': len(self.memory()['skills']),
                'local_only': True, 'kernel_replaced': False, 'human_level_intelligence': False}

    def _safe_workspace_path(self, relative):
        raw = Path(relative)
        if raw.is_absolute() or not relative or '\\' in relative and os.name != 'nt':
            raise ValueError('Only workspace-relative paths are accepted')
        dest = (self.workspace / raw).resolve()
        if not dest.is_relative_to(self.workspace):
            raise ValueError('Access outside Axiom workspace denied')
        return dest

    def observe_file(self, relative):
        target = self._safe_workspace_path(relative)
        if not target.is_file():
            raise FileNotFoundError('File not found inside Axiom workspace')
        if target.stat().st_size > 2_000_000:
            raise ValueError('File exceeds observation limit')
        data = target.read_bytes()
        decoded = data.decode('utf-8', errors='replace')
        evidence = {'file': str(target.relative_to(self.workspace)), 'bytes': len(data),
                    'characters': len(decoded), 'sha256': hashlib.sha256(data).hexdigest()}
        return self._event('perception', relative, Result('perception', 'OBSERVED',
                            f"Observed {evidence['bytes']} bytes / {evidence['characters']} characters.",
                            evidence, True))

    def write_file(self, relative, content, confirmed=False):
        if not confirmed:
            return Result('action', 'APPROVAL_REQUIRED', 'Writing files requires an explicit confirmed action.',
                          {'path': relative}, False)
        target = self._safe_workspace_path(relative)
        if len(content.encode('utf-8')) > 200_000:
            raise ValueError('File content exceeds safe limit')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        observed = self.observe_file(relative)
        return self._event('action', f'write {relative}', Result('action', 'WRITTEN',
                           f'Wrote and checked {relative}', observed.evidence, True))

    def goal(self, intent):
        if not intent.strip():
            raise ValueError('Goal is empty')
        with self._lock, self.connection:
            row = self.connection.execute('INSERT INTO goals(intent,state,created) VALUES(?,?,?)',
                                          (intent[:2000], 'RESEARCH', utcnow()))
            gid = row.lastrowid
        # Do not fabricate a solution by interpreting the goal as already achieved.
        plan = {'goal_id': gid, 'intent': intent[:2000], 'stages': [
            'Identify observable success criteria',
            'Enumerate candidate approaches and required capabilities',
            'Implement an isolated experiment',
            'Test the result against success criteria',
            'Preserve evidence and revise if needed'],
            'status': 'RESEARCH', 'automatic_deep_analysis': True}
        return self._event('goal', intent, Result('goal', 'PLANNED',
                           'Research plan created; no implementation or success claimed.', plan, False))

    def generate_local(self, prompt, max_tokens=96):
        """Use the separately trained original byte model when installed.

        Unlike symbolic results, generated language has no verified truth guarantee.
        """
        try:
            from continuum.charllm import complete, DEFAULT_CKPT
        except ImportError:
            return self._event('generation', prompt, Result('generation', 'MODEL_NOT_INSTALLED',
                'The original Axiom byte model is not in this installation.', {}, False))
        if not DEFAULT_CKPT.is_file():
            return self._event('generation', prompt, Result('generation', 'CHECKPOINT_MISSING',
                'Train the Axiom byte model or copy its own model checkpoint.', {}, False))
        try:
            text = complete('Teacher: ' + prompt + '\nStudent: ', max_new_tokens=max_tokens, seed=7)
        except (ValueError, RuntimeError, OSError) as exc:
            return self._event('generation', prompt, Result('generation', 'GENERATION_FAILED',
                str(exc), {'model': 'original-byte-transformer'}, False))
        return self._event('generation', prompt, Result('generation', 'UNVERIFIED_GENERATION',
            text, {'model': 'original-byte-transformer', 'max_tokens': max_tokens,
                   'note': 'Generated text is not verified reasoning.'}, False))

    def events(self, limit=30):
        with self._lock:
            return [dict(row) for row in self.connection.execute(
                'SELECT id,at,kind,input,status,evidence FROM events ORDER BY id DESC LIMIT ?',
                (min(max(int(limit), 1), 200),)).fetchall()]

    def respond(self, message):
        """Transparent deterministic interface; never misrepresent as a general LLM."""
        line = message.strip()
        if not line:
            return Result('dialogue', 'EMPTY', 'Enter a goal, skill, or observation.', {}, False)
        chunks = line.split(' ', 2)
        verb = chunks[0].lower()
        try:
            if verb == 'generate':
                return self.generate_local(line[len('generate'):].strip())
            if verb == 'status':
                s = self.status()
                return Result('perception', 'OBSERVED', json.dumps(s, indent=2), s, True)
            if verb == 'goal':
                return self.goal(line[5:].strip())
            if verb == 'remember' and len(chunks) == 3:
                return self.remember(chunks[1], chunks[2])
            if verb == 'recall':
                q = line[len('recall'):].strip()
                hits = self.recall(q)
                return Result('memory', 'RETRIEVED', json.dumps(hits, indent=2), {'count':len(hits)}, True)
            if verb == 'observe':
                return self.observe_file(line[len('observe'):].strip())
            if verb == 'teach':
                match = re.fullmatch(r'teach\s+([A-Za-z][A-Za-z0-9_]*)\s+(.+?)\s*=>\s*(.*)', line, flags=re.S | re.I)
                if match:
                    return self.teach(match.group(1), match.group(2), match.group(3))
                return Result('learning', 'FORMAT', 'Use: teach SKILL INPUT => OUTPUT', {}, False)
            if verb == 'invent' and len(chunks) == 2:
                return self.invent(chunks[1])
            if verb == 'perform' and len(chunks) == 3:
                return self.perform(chunks[1], chunks[2])
            if verb == 'verify':
                match = re.fullmatch(r'verify\s+([A-Za-z][A-Za-z0-9_]*)\s+(.+?)\s*=>\s*(.*)', line, flags=re.S | re.I)
                if match:
                    return self.verify(match.group(1), match.group(2), match.group(3))
                return Result('verification', 'FORMAT', 'Use: verify SKILL INPUT => OUTPUT', {}, False)
            if verb == 'evidence':
                rows = self.events()
                return Result('evidence', 'RETRIEVED', json.dumps(rows, indent=2), {'count':len(rows)}, True)
            if verb == 'help':
                return Result('dialogue', 'SUPPORTED', 'Commands: generate ... | status | goal ... | teach SKILL INPUT => OUTPUT | invent SKILL | perform SKILL VALUE | verify SKILL INPUT => OUTPUT | remember NAME TEXT | recall WORD | observe FILE | evidence', {}, True)
            # Default to serious goal analysis rather than depending on a trigger word.
            return self.goal(line)
        except (ValueError, vm.MachineError, OSError) as exc:
            return self._event('diagnostic', line, Result('diagnostic', 'ACTION_NOT_COMPLETED',
                               str(exc), {'exception_type': type(exc).__name__}, False))