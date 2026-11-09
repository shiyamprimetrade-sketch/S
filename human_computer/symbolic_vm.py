"""Axiom Programmable Computer v0.1.

An offline, standard-library-only symbolic stack virtual machine.
It can assemble NEW operations by searching compositions of documented primitives
against teacher examples, then run/test/save the discovered program.
Not an OS replacement, hardware modification, LLM, or arbitrary invention system.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import tempfile

SCHEMA = 1
MAX_TEXT = 4096
MAX_STACK = 8
INSTRUCTIONS = (
    'DUP', 'SWAP', 'JOIN', 'REVERSE', 'UPPER', 'LOWER',
    'LEN', 'TO_WORD', 'STR', 'DOUBLE', 'FIRST', 'LAST',
    'SORT', 'REMOVE_SPACES', 'TRIM'
)


class MachineError(Exception):
    """A recognized condition returned with a useful explanation."""


def number_word(n: int) -> str:
    ones = ('ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN',
            'EIGHT', 'NINE', 'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN',
            'FIFTEEN', 'SIXTEEN', 'SEVENTEEN', 'EIGHTEEN', 'NINETEEN')
    tens = ('', '', 'TWENTY', 'THIRTY', 'FORTY', 'FIFTY', 'SIXTY',
            'SEVENTY', 'EIGHTY', 'NINETY')
    if not isinstance(n, int) or not 0 <= n <= 9999:
        raise MachineError('TO_WORD currently accepts integers from 0 to 9999')
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + (ones[n % 10] if n % 10 else '')
    if n < 1000:
        return ones[n // 100] + 'HUNDRED' + (number_word(n % 100) if n % 100 else '')
    return ones[n // 1000] + 'THOUSAND' + (number_word(n % 1000) if n % 1000 else '')


def operation(stack: tuple, opcode: str) -> tuple:
    """Apply one pure, bounded instruction. No shell or filesystem access."""
    if opcode not in INSTRUCTIONS:
        raise MachineError(f'unknown instruction {opcode!r}; use instructions')
    values = list(stack)
    if not values:
        raise MachineError('empty stack')
    if opcode == 'DUP':
        if len(values) >= MAX_STACK:
            raise MachineError('stack capacity reached')
        values.append(values[-1])
    elif opcode == 'SWAP':
        if len(values) < 2:
            raise MachineError('SWAP needs two values')
        values[-1], values[-2] = values[-2], values[-1]
    elif opcode == 'JOIN':
        if len(values) < 2 or not isinstance(values[-1], str) or not isinstance(values[-2], str):
            raise MachineError('JOIN needs two strings')
        right, left = values.pop(), values.pop()
        values.append(left + right)
    else:
        v = values.pop()
        text_ops = {
            'REVERSE': lambda s: s[::-1], 'UPPER': str.upper, 'LOWER': str.lower,
            'FIRST': lambda s: s[:1], 'LAST': lambda s: s[-1:],
            'SORT': lambda s: ''.join(sorted(s)),
            'REMOVE_SPACES': lambda s: s.replace(' ', ''), 'TRIM': str.strip
        }
        if opcode in text_ops:
            if not isinstance(v, str):
                raise MachineError(f'{opcode} needs text')
            values.append(text_ops[opcode](v))
        elif opcode == 'LEN':
            if not isinstance(v, str):
                raise MachineError('LEN needs text')
            values.append(len(v))
        elif opcode == 'TO_WORD':
            values.append(number_word(v))
        elif opcode == 'STR':
            values.append(str(v))
        elif opcode == 'DOUBLE':
            if type(v) is not int:
                raise MachineError('DOUBLE needs an integer')
            values.append(v * 2)
    if any(isinstance(v, str) and len(v) > MAX_TEXT for v in values):
        raise MachineError(f'text limit {MAX_TEXT} characters exceeded')
    return tuple(values)


def execute(program: list[str] | tuple[str, ...], value: str, with_trace=False) -> dict:
    if len(value) > MAX_TEXT:
        raise MachineError(f'input exceeds {MAX_TEXT} characters')
    if len(program) > 16:
        raise MachineError('program exceeds 16 instruction safety ceiling')
    stack = (value,)
    history = [{'instruction': 'INPUT', 'stack': list(stack)}]
    for code in program:
        stack = operation(stack, code)
        if with_trace:
            history.append({'instruction': code, 'stack': list(stack)})
    if len(stack) != 1:
        raise MachineError(f'program ended with {len(stack)} values; needs exactly one')
    result = {'output': str(stack[0])}
    if with_trace:
        result['trace'] = history
    return result


def empty_memory() -> dict:
    return {'schema': SCHEMA, 'skills': {}}


def read_memory(path: Path) -> dict:
    if not path.exists():
        return empty_memory()
    m = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(m, dict) or m.get('schema') != SCHEMA or not isinstance(m.get('skills'), dict):
        raise MachineError('invalid memory file; not overwritten')
    return m


def save_memory(path: Path, memory: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.axiom-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def teach(memory: dict, skill: str, original: str, expected: str) -> dict:
    if not skill.isidentifier() or len(skill) > 64:
        raise MachineError('skill name must contain letters, numbers, underscore; start with a letter')
    if len(original) > MAX_TEXT or len(expected) > MAX_TEXT:
        raise MachineError('example exceeds text limit')
    skills = memory['skills']
    record = skills.setdefault(skill, {'examples': {}, 'program': None, 'tests': [], 'revision': 0})
    previous = record['examples'].get(original)
    if previous is not None and previous != expected:
        return {'status': 'CONFLICT', 'previous': previous, 'proposed': expected,
                'next': 'Use correct to explicitly revise a conflicting example.'}
    if previous == expected:
        return {'status': 'UNCHANGED', 'examples': len(record['examples'])}
    record['examples'][original] = expected
    record['program'] = None  # new evidence requires re-induction
    record['tests'] = []
    record['revision'] += 1
    return {'status': 'LEARNED_EXAMPLE', 'examples': len(record['examples']),
            'next': 'Provide another example and run invent.'}


def correct(memory: dict, skill: str, original: str, expected: str) -> dict:
    if skill not in memory['skills'] or original not in memory['skills'][skill]['examples']:
        raise MachineError('nothing to correct; use teach first')
    record = memory['skills'][skill]
    if record['examples'][original] == expected:
        return {'status': 'UNCHANGED'}
    record['examples'][original] = expected
    record['program'] = None
    record['tests'] = []
    record['revision'] += 1
    return {'status': 'CORRECTED', 'revision': record['revision'], 'next': 'Run invent again.'}


def synthesize(examples: dict[str, str], max_depth: int = 5, max_states: int = 60000) -> dict:
    """Breadth-first search for a minimal valid composition, not open-ended AI synthesis."""
    if len(examples) < 2:
        return {'status': 'MORE_EXAMPLES_REQUIRED', 'next': 'Teach at least two distinct inputs.'}
    if not 1 <= max_depth <= 7:
        raise MachineError('search depth must be from 1 through 7')
    start = tuple((key,) for key in examples)
    wanted = tuple(examples.values())
    queue = deque([((), start)])
    seen = {start}
    processed = 0
    matches = []
    solution_depth = None
    while queue and processed < max_states:
        path, states = queue.popleft()
        processed += 1
        if solution_depth is not None and len(path) >= solution_depth:
            continue
        if len(path) >= max_depth:
            continue
        for instruction in INSTRUCTIONS:
            try:
                transformed = tuple(operation(st, instruction) for st in states)
            except MachineError:
                continue
            new_path = path + (instruction,)
            if all(len(st) == 1 and str(st[0]) == expected
                   for st, expected in zip(transformed, wanted)):
                matches.append(new_path)
                solution_depth = len(new_path)
                if len(matches) >= 10:
                    break
                continue
            if solution_depth is None and transformed not in seen:
                seen.add(transformed)
                queue.append((new_path, transformed))
        if len(matches) >= 10:
            break
    if matches:
        return {'status': 'CANDIDATE_FOUND', 'program': list(matches[0]),
                'alternatives': [list(p) for p in matches[1:]],
                'searched_states': processed, 'minimal_instructions': len(matches[0]),
                'evidence': 'Fits teaching examples only; verify on unseen input.'}
    return {'status': 'SEARCH_LIMIT' if processed >= max_states else 'NO_PROGRAM_FOUND',
            'searched_states': processed,
            'next': 'Try more teaching examples, deeper search or a new primitive instruction.'}


def invent(memory: dict, skill: str, depth: int = 5) -> dict:
    record = memory['skills'].get(skill)
    if record is None:
        raise MachineError('unknown skill; teach first')
    finding = synthesize(record['examples'], max_depth=depth)
    if finding['status'] == 'CANDIDATE_FOUND':
        record['program'] = finding['program']
        record['tests'] = []
        record['revision'] += 1
        finding['skill'] = skill
        finding['next'] = f'Run and verify this program on a new input: prove {skill} INPUT EXPECTED'
    return finding


def define(memory: dict, skill: str, codes: list[str]) -> dict:
    if not skill.isidentifier() or len(skill) > 64:
        raise MachineError('invalid skill name')
    if not codes or len(codes) > 16 or any(c not in INSTRUCTIONS for c in codes):
        raise MachineError('1–16 known instructions required; use instructions to list')
    # Validate entire sequence against sample input; typed input may need testing later.
    skills = memory['skills']
    record = skills.setdefault(skill, {'examples': {}, 'program': None, 'tests': [], 'revision': 0})
    for inp, expected in record['examples'].items():
        got = execute(codes, inp)['output']
        if got != expected:
            return {'status': 'MISMATCH', 'input': inp, 'expected': expected, 'actual': got}
    record['program'] = codes
    record['tests'] = []
    record['revision'] += 1
    return {'status': 'REPROGRAMMED', 'program': codes,
            'next': f'Run {skill} on a new input and prove the result.'}


def run(memory: dict, skill: str, value: str, show_trace=False) -> dict:
    record = memory['skills'].get(skill)
    if not record or not record['program']:
        return {'status': 'NOT_PROGRAMMED', 'next': 'Teach examples and run invent, or use define.'}
    res = execute(record['program'], value, with_trace=show_trace)
    recent = [t for t in record['tests'] if t['revision'] == record['revision']]
    passed = sum(t['passed'] for t in recent)
    failed = sum(not t['passed'] for t in recent)
    return {'status': 'OUTPUT', 'skill': skill, 'input': value, 'output': res['output'],
            'program': record['program'], 'evidence': {'passed_unseen': passed, 'failed_unseen': failed,
            'label': 'FAILED_TEST_PRESENT' if failed else ('TESTED' if passed else 'UNTESTED')},
            **({'trace': res['trace']} if show_trace else {})}


def prove(memory: dict, skill: str, value: str, expected: str) -> dict:
    record = memory['skills'].get(skill)
    if not record or not record['program']:
        return {'status': 'NOT_PROGRAMMED'}
    if value in record['examples']:
        return {'status': 'NOT_UNSEEN', 'next': 'Use a different input not supplied during teaching.'}
    result = run(memory, skill, value)
    passed = result['output'] == expected
    record['tests'].append({'input': value, 'actual': result['output'], 'expected': expected,
                            'passed': passed, 'revision': record['revision'],
                            'time': datetime.now(timezone.utc).isoformat()})
    return {'status': 'TEST_PASS' if passed else 'TEST_FAIL', 'actual': result['output'],
            'expected': expected, 'meaning': 'Evidence on one new input; not universal proof.'}


def demo() -> list[dict]:
    memory = empty_memory()
    actions = [teach(memory, 'mirror', 'CAT', 'CATTAC'),
               teach(memory, 'mirror', 'DOG', 'DOGGOD'),
               invent(memory, 'mirror', depth=4),
               run(memory, 'mirror', 'SUN', show_trace=True),
               prove(memory, 'mirror', 'SUN', 'SUNNUS')]
    return actions


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Axiom programmable symbolic computer, offline')
    p.add_argument('--memory', default='axiom_computer_memory.json')
    sub = p.add_subparsers(dest='command', required=True)
    for command, names in {
        'teach': ('skill', 'input', 'output'), 'correct': ('skill', 'input', 'output'),
        'invent': ('skill',), 'define': ('skill', 'instructions'),
        'run': ('skill', 'input'), 'trace': ('skill', 'input'),
        'prove': ('skill', 'input', 'expected'), 'inspect': ('skill',)
    }.items():
        q = sub.add_parser(command)
        for name in names:
            if name == 'instructions':
                q.add_argument(name, nargs='+')
            else:
                q.add_argument(name)
        if command == 'invent':
            q.add_argument('--depth', type=int, default=5)
    for command in ('demo', 'list', 'shell', 'instructions'):
        sub.add_parser(command)
    return p


def dispatch(args, memory: dict) -> tuple[dict, bool]:
    cmd = args.command
    if cmd == 'teach':
        out = teach(memory, args.skill, args.input, args.output)
        return out, out['status'] != 'CONFLICT'
    if cmd == 'correct':
        return correct(memory, args.skill, args.input, args.output), True
    if cmd == 'invent':
        out = invent(memory, args.skill, args.depth)
        return out, out['status'] == 'CANDIDATE_FOUND'
    if cmd == 'define':
        out = define(memory, args.skill, args.instructions)
        return out, out['status'] == 'REPROGRAMMED'
    if cmd == 'run' or cmd == 'trace':
        return run(memory, args.skill, args.input, show_trace=cmd == 'trace'), False
    if cmd == 'prove':
        out = prove(memory, args.skill, args.input, args.expected)
        return out, out['status'] in ('TEST_PASS', 'TEST_FAIL')
    if cmd == 'inspect':
        return memory['skills'].get(args.skill, {'status': 'NOT_FOUND'}), False
    if cmd == 'list':
        return {'skills': {k: {'examples': len(v['examples']), 'program': v['program'],
                              'tests': len(v['tests'])} for k, v in memory['skills'].items()}}, False
    if cmd == 'instructions':
        return {'instructions': INSTRUCTIONS, 'notes': 'All operations are local and bounded.'}, False
    if cmd == 'demo':
        return {'demo': demo()}, False
    raise MachineError('unsupported command')


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    path = Path(args.memory)
    try:
        if args.command == 'shell':
            print('AXIOM COMPUTER — type help, or exit. Example: teach mirror CAT CATTAC')
            while True:
                try:
                    line = input('axiom> ').strip()
                except (KeyboardInterrupt, EOFError):
                    print('\nSession ended. Memory saved on each change.')
                    return 0
                if line.lower() in ('exit', 'quit'):
                    return 0
                if line.lower() in ('help', '?'):
                    p.print_help()
                    continue
                if not line:
                    continue
                try:
                    fragments = shlex.split(line)
                    if fragments[0] == 'shell':
                        print('Already in the interactive shell.')
                        continue
                    status = main(['--memory', str(path), *fragments])
                    if status:
                        print('Use help to inspect valid commands.')
                except (ValueError, SystemExit) as exc:
                    print(f'Input inspection needed: {exc}')
        memory = read_memory(path)
        result, changed = dispatch(args, memory)
        if changed:
            save_memory(path, memory)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (MachineError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'INSPECTION_NEEDED', 'details': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())