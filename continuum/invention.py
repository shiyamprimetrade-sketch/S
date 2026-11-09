"""S Invention Lab: teacher-supervised, evidence-first symbolic program synthesis.

This is deliberately a bounded symbolic learner, NOT autonomous general intelligence.
No generated Python, shell commands, network activity, or host file writes are executed.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .storage import Store, new_id, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS inventions (
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 objective TEXT NOT NULL, success_criteria TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'awaiting_examples', candidate TEXT,
 version INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inventions_project ON inventions(project_id, created_at);
CREATE TABLE IF NOT EXISTS invention_examples (
 id TEXT PRIMARY KEY, invention_id TEXT NOT NULL REFERENCES inventions(id) ON DELETE CASCADE,
 input TEXT NOT NULL, expected TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('teach','test')),
 teacher_note TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_invention_examples ON invention_examples(invention_id, role);
CREATE TABLE IF NOT EXISTS invention_runs (
 id TEXT PRIMARY KEY, invention_id TEXT NOT NULL REFERENCES inventions(id) ON DELETE CASCADE,
 version INTEGER NOT NULL, candidate TEXT NOT NULL, passed INTEGER NOT NULL,
 results_json TEXT NOT NULL, evidence_sha256 TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_invention_runs ON invention_runs(invention_id,created_at);
"""

# A small, auditable instruction vocabulary. These are actual functions, not labels.
OPERATIONS: dict[str, Callable[[str], str]] = {
    'identity': lambda s: s,
    'reverse': lambda s: s[::-1],
    'uppercase': lambda s: s.upper(),
    'lowercase': lambda s: s.lower(),
    'duplicate': lambda s: s + s,
    'append_s': lambda s: s + 'S',
    'prepend_s': lambda s: 'S' + s,
    'append_reverse': lambda s: s + s[::-1],
    'reverse_append': lambda s: s[::-1] + s,
    'length_digits': lambda s: str(len(s)),
    'first': lambda s: s[:1],
    'last': lambda s: s[-1:],
}


def execute(program: list[str], text: str) -> str:
    """Execute at most two allowed transformations, never arbitrary user code."""
    if not isinstance(program, list) or not 1 <= len(program) <= 2:
        raise ValueError('Only one- or two-step symbolic programs are supported')
    if not isinstance(text, str) or len(text) > 500:
        raise ValueError('Input must be text of at most 500 characters')
    for op in program:
        if op not in OPERATIONS:
            raise ValueError('Unknown or unapproved instruction')
        text = OPERATIONS[op](text)
        if len(text) > 2000:
            raise ValueError('Program exceeded output size limit')
    return text


def candidates():
    """Prefer simple programs; deduplicate paths that do identical things by evaluation."""
    for name in OPERATIONS:
        yield [name]
    for a in OPERATIONS:
        for b in OPERATIONS:
            if a != 'identity' and b != 'identity':
                yield [a, b]


class InventionLab:
    def __init__(self, store: Store):
        self.store = store
        with store.connect() as db:
            db.executescript(SCHEMA)

    def _goal(self, project_id, goal_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            row = db.execute('SELECT * FROM inventions WHERE id=? AND project_id=?',
                             (goal_id, project_id)).fetchone()
        if not row:
            raise ValueError('Unknown invention for this project')
        return dict(row)

    def create(self, project_id, objective, success_criteria):
        self.store.get_project(project_id)
        objective = str(objective).strip()
        success_criteria = str(success_criteria).strip()
        if not 3 <= len(objective) <= 1000 or not 3 <= len(success_criteria) <= 1000:
            raise ValueError('Objective and observable success criteria must be 3–1000 characters')
        goal = dict(id=new_id(), project_id=project_id, objective=objective,
                    success_criteria=success_criteria, status='awaiting_examples', candidate=None,
                    version=0, created_at=now(), updated_at=now())
        with self.store.connect() as db:
            db.execute('INSERT INTO inventions VALUES (:id,:project_id,:objective,:success_criteria,'
                       ':status,:candidate,:version,:created_at,:updated_at)', goal)
        self.store.log(project_id, 'invention-created', f"goal={goal['id']}")
        return goal

    def list(self, project_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            return [dict(row) for row in db.execute(
                'SELECT * FROM inventions WHERE project_id=? ORDER BY rowid DESC', (project_id,))]

    def detail(self, project_id, goal_id):
        goal = self._goal(project_id, goal_id)
        with self.store.connect() as db:
            goal['examples'] = [dict(r) for r in db.execute(
                'SELECT * FROM invention_examples WHERE invention_id=? ORDER BY rowid', (goal_id,))]
            goal['runs'] = [dict(r) for r in db.execute(
                'SELECT * FROM invention_runs WHERE invention_id=? ORDER BY rowid DESC LIMIT 30', (goal_id,))]
        for run in goal['runs']:
            run['results'] = json.loads(run.pop('results_json'))
            run['passed'] = bool(run['passed'])
        goal['candidate'] = json.loads(goal['candidate']) if goal['candidate'] else None
        return goal

    def _invalidate(self, project_id, goal_id, db):
        db.execute("UPDATE inventions SET candidate=NULL,status='awaiting_synthesis',"
                   'version=version+1,updated_at=? WHERE id=? AND project_id=?',
                   (now(), goal_id, project_id))

    def add_example(self, project_id, goal_id, input_text, expected, role='teach'):
        self._goal(project_id, goal_id)
        if role not in ('teach', 'test'):
            raise ValueError('Example role must be teach or test')
        if not isinstance(input_text, str) or not isinstance(expected, str):
            raise ValueError('Examples must be strings')
        if not 1 <= len(input_text) <= 500 or len(expected) > 2000:
            raise ValueError('Invalid example length')
        item = dict(id=new_id(), invention_id=goal_id, input=input_text,
                    expected=expected, role=role, teacher_note='', revision=0, created_at=now())
        with self.store.connect() as db:
            # Never leak holdout labels into the synthesis training set.
            exists = db.execute('SELECT id FROM invention_examples WHERE invention_id=? AND input=?',
                                (goal_id, input_text)).fetchone()
            if exists:
                raise ValueError('Duplicate input: use correction to change an existing example')
            db.execute('INSERT INTO invention_examples VALUES (:id,:invention_id,:input,:expected,'
                       ':role,:teacher_note,:revision,:created_at)', item)
            self._invalidate(project_id, goal_id, db)
        self.store.log(project_id, 'teaching-example', f"goal={goal_id} example={item['id']} role={role}")
        return item

    def correct(self, project_id, goal_id, example_id, expected, teacher_note):
        self._goal(project_id, goal_id)
        if not isinstance(expected, str) or len(expected) > 2000:
            raise ValueError('Invalid corrected output')
        if not isinstance(teacher_note, str) or not 1 <= len(teacher_note.strip()) <= 1000:
            raise ValueError('Teacher correction requires an explanation')
        with self.store.connect() as db:
            row = db.execute('SELECT id FROM invention_examples WHERE id=? AND invention_id=?',
                             (example_id, goal_id)).fetchone()
            if not row:
                raise ValueError('Unknown example')
            db.execute('UPDATE invention_examples SET expected=?,teacher_note=?,revision=revision+1 WHERE id=?',
                       (expected, teacher_note.strip(), example_id))
            self._invalidate(project_id, goal_id, db)
        self.store.log(project_id, 'teacher-correction', f"goal={goal_id} example={example_id} revision")
        # Explicit correction triggers a new attempt. An incompatible lesson remains unresolved,
        # not silently treated as success.
        return self.synthesize(project_id, goal_id)

    def synthesize(self, project_id, goal_id):
        self._goal(project_id, goal_id)
        with self.store.connect() as db:
            rows = [dict(r) for r in db.execute('SELECT input,expected FROM invention_examples '
                        "WHERE invention_id=? AND role='teach' ORDER BY rowid", (goal_id,))]
        if len(rows) < 2:
            raise ValueError('Teach at least two distinct examples before synthesis')
        valid = []
        for program in candidates():
            try:
                if all(execute(program, r['input']) == r['expected'] for r in rows):
                    valid.append(program)
            except ValueError:
                pass
        candidate = valid[0] if valid else None
        status = 'hypothesis' if candidate else 'unresolved'
        with self.store.connect() as db:
            db.execute('UPDATE inventions SET status=?,candidate=?,updated_at=? WHERE id=? AND project_id=?',
                       (status, json.dumps(candidate) if candidate else None, now(), goal_id, project_id))
        self.store.log(project_id, 'symbolic-synthesis',
                       f'goal={goal_id} status={status} candidates={len(valid)}')
        return {'goal_id': goal_id, 'status': status, 'candidate': candidate,
                'consistent_candidates': len(valid),
                'interpretation': 'bounded symbolic hypothesis, not human-level understanding'}

    def predict(self, project_id, goal_id, input_text):
        goal = self._goal(project_id, goal_id)
        if not goal['candidate']:
            raise ValueError('No current candidate: teach or revise examples and synthesize')
        program = json.loads(goal['candidate'])
        return {'input': input_text, 'output': execute(program, input_text),
                'candidate': program, 'status': goal['status'],
                'claim': 'symbolic candidate prediction; not universal correctness'}

    def evaluate(self, project_id, goal_id):
        goal = self._goal(project_id, goal_id)
        if not goal['candidate']:
            raise ValueError('No candidate to evaluate')
        with self.store.connect() as db:
            tests = [dict(r) for r in db.execute('SELECT id,input,expected FROM invention_examples '
                       "WHERE invention_id=? AND role='test' ORDER BY rowid", (goal_id,))]
        if not tests:
            raise ValueError('At least one independent held-out test is required')
        program = json.loads(goal['candidate'])
        results = []
        for case in tests:
            try:
                actual = execute(program, case['input'])
                passed = actual == case['expected']
                results.append(dict(example_id=case['id'], input=case['input'], expected=case['expected'],
                                    actual=actual, passed=passed))
            except ValueError as e:
                results.append(dict(example_id=case['id'], input=case['input'], expected=case['expected'],
                                    actual=None, passed=False, error=str(e)))
        passed = all(item['passed'] for item in results)
        evidence = {'program': program, 'version': goal['version'], 'tests': results}
        body = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
        run = dict(id=new_id(), invention_id=goal_id, version=goal['version'],
                   candidate=json.dumps(program), passed=int(passed), results_json=json.dumps(results),
                   evidence_sha256=hashlib.sha256(body.encode()).hexdigest(), created_at=now())
        status = 'verified_on_heldout_cases' if passed else 'falsified_by_heldout_cases'
        with self.store.connect() as db:
            db.execute('INSERT INTO invention_runs VALUES (:id,:invention_id,:version,:candidate,'
                       ':passed,:results_json,:evidence_sha256,:created_at)', run)
            db.execute('UPDATE inventions SET status=?,updated_at=? WHERE id=? AND project_id=?',
                       (status, now(), goal_id, project_id))
        self.store.log(project_id, 'symbolic-experiment',
                       f'goal={goal_id} run={run["id"]} status={status} evidence_sha256={run["evidence_sha256"]}')
        return {'run_id': run['id'], 'goal_id': goal_id, 'candidate': program, 'status': status,
                'passed': passed, 'results': results, 'evidence_sha256': run['evidence_sha256'],
                'limits': 'Only these held-out cases were tested. This does not establish general intelligence.'}
