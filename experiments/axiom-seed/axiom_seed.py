"""Axiom Seed v0.1: one-command, example-taught symbolic transformations.

Offline, standard-library-only, transparent rule induction, local memory.
This is a small learning prototype, NOT a general language model.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def number_word(n: int) -> str:
    """Return English word for 0..9999 (without spaces/hyphens)."""
    ones = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE',
            'TEN', 'ELEVEN', 'TWELVE', 'THIRTEEN', 'FOURTEEN', 'FIFTEEN', 'SIXTEEN',
            'SEVENTEEN', 'EIGHTEEN', 'NINETEEN']
    tens = ['', '', 'TWENTY', 'THIRTY', 'FORTY', 'FIFTY', 'SIXTY', 'SEVENTY', 'EIGHTY', 'NINETY']
    if not 0 <= n <= 9999:
        raise ValueError('number words supported for lengths from 0 through 9999')
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + (ones[n % 10] if n % 10 else '')
    if n < 1000:
        return ones[n // 100] + 'HUNDRED' + (number_word(n % 100) if n % 100 else '')
    return ones[n // 1000] + 'THOUSAND' + (number_word(n % 1000) if n % 1000 else '')


def apply(name: str, value: str) -> str:
    transforms = {
        'identity': lambda s: s,
        'uppercase': lambda s: s.upper(),
        'lowercase': lambda s: s.lower(),
        'titlecase': lambda s: s.title(),
        'reverse': lambda s: s[::-1],
        'trim': lambda s: s.strip(),
        'remove_spaces': lambda s: s.replace(' ', ''),
        'sort_letters': lambda s: ''.join(sorted(s)),
        'first_character': lambda s: s[:1],
        'last_character': lambda s: s[-1:],
        'count_characters': lambda s: str(len(s)),
        'count_as_word': lambda s: number_word(len(s)),
    }
    return transforms[name](value)


RULES = ('identity', 'uppercase', 'lowercase', 'titlecase', 'reverse', 'trim',
         'remove_spaces', 'sort_letters', 'first_character', 'last_character',
         'count_characters', 'count_as_word')


def new_memory() -> dict:
    return {'schema': 1, 'skills': {}}


def read_memory(path: Path) -> dict:
    if not path.exists():
        return new_memory()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema') != 1 or not isinstance(data.get('skills'), dict):
        raise ValueError('invalid memory schema; original file was not modified')
    return data


def save_memory(path: Path, memory: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix='.axiom-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(memory, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def matching_rules(examples: dict) -> list[str]:
    if not examples:
        return list(RULES)
    candidates = []
    for name in RULES:
        try:
            if all(apply(name, x) == y for x, y in examples.items()):
                candidates.append(name)
        except (ValueError, TypeError):
            pass
    return candidates


def skill_record(memory: dict, name: str) -> dict:
    skills = memory['skills']
    if name not in skills:
        skills[name] = {'examples': {}, 'tests': [], 'revision': 0}
    return skills[name]


def teach(memory: dict, name: str, original: str, expected: str, correct: bool = False) -> dict:
    skill = skill_record(memory, name)
    old = skill['examples'].get(original)
    if old is not None and old != expected and not correct:
        return {'status': 'CONFLICT', 'details': 'Existing example disagrees. Use correct to explicitly replace it.',
                'previous': old, 'proposed': expected}
    if old != expected:
        skill['examples'][original] = expected
        skill['revision'] += 1
    candidates = matching_rules(skill['examples'])
    return {'status': 'READY_TO_TEST' if len(candidates) == 1 and len(skill['examples']) >= 2 else
            ('NEEDS_MORE_EXAMPLES' if candidates else 'NO_MATCHING_RULE'),
            'examples': len(skill['examples']), 'candidate_rules': candidates,
            'message': ('The 12 built-in hypothesis functions cannot explain these examples.' if not candidates else
                        'Try an unseen input and verify the prediction.' if len(candidates) == 1 and len(skill['examples']) >= 2 else
                        'Supply another distinct example to narrow possible rules.')}


def predict(memory: dict, name: str, value: str) -> dict:
    skill = memory['skills'].get(name)
    if not skill:
        return {'status': 'NEEDS_TEACHING', 'message': 'Skill not created; teach at least two examples.'}
    examples = skill['examples']
    candidates = matching_rules(examples)
    if len(examples) < 2 or len(candidates) != 1:
        return {'status': 'NEEDS_MORE_EXAMPLES' if candidates else 'NO_MATCHING_RULE',
                'candidate_rules': candidates,
                'message': 'A unique rule has not yet been established.'}
    rule = candidates[0]
    try:
        result = apply(rule, value)
    except ValueError as exc:
        return {'status': 'UNSUPPORTED_INPUT', 'message': str(exc)}
    tests = [t for t in skill['tests'] if t['revision'] == skill['revision']]
    if any(not t['passed'] for t in tests):
        confidence = 'UNDER_REVIEW_FAILED_TEST'
    elif any(t['passed'] for t in tests):
        confidence = 'HAS_PASSED_UNSEEN_TEST'
    else:
        confidence = 'UNTESTED_GENERALIZATION'
    return {'status': 'PREDICTION', 'input': value, 'output': result, 'rule': rule,
            'evidence': confidence, 'examples': len(examples),
            'tested_input': value in examples}


def prove(memory: dict, name: str, value: str, expected: str) -> dict:
    skill = memory['skills'].get(name)
    if not skill:
        return {'status': 'NEEDS_TEACHING'}
    if value in skill['examples']:
        return {'status': 'NOT_UNSEEN', 'message': 'Choose an input not already in teaching examples.'}
    proposed = predict(memory, name, value)
    if proposed['status'] != 'PREDICTION':
        return proposed
    passed = proposed['output'] == expected
    skill['tests'].append({'input': value, 'expected': expected, 'actual': proposed['output'],
                           'passed': passed, 'revision': skill['revision'],
                           'timestamp': datetime.now(timezone.utc).isoformat()})
    return {'status': 'TEST_PASS' if passed else 'TEST_FAIL',
            'input': value, 'predicted': proposed['output'], 'expected': expected,
            'interpretation': ('Supports this rule on one new input, not universal proof.' if passed else
                               'The rule failed to generalize; more teaching or a new rule is needed.')}


def trace(memory: dict, name: str, value: str, max_steps: int = 12) -> dict:
    seen = {}
    sequence = []
    current = value
    for _ in range(max_steps):
        if current in seen:
            return {'status': 'CYCLE', 'sequence': sequence + [current], 'cycle_starts_at': seen[current]}
        seen[current] = len(sequence)
        sequence.append(current)
        next_value = predict(memory, name, current)
        if next_value['status'] != 'PREDICTION':
            return {'status': next_value['status'], 'sequence': sequence}
        current = next_value['output']
    return {'status': 'STEP_LIMIT', 'sequence': sequence, 'note': 'Stopped to prevent infinite work.'}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Axiom Seed: learn a tiny symbolic skill from examples, offline.')
    parser.add_argument('--memory', default='axiom_memory.json', help='Local persistent learning file')
    sub = parser.add_subparsers(dest='command', required=True)
    for command, names in {'teach': ('skill', 'input', 'output'), 'correct': ('skill', 'input', 'output'),
                           'run': ('skill', 'input'), 'prove': ('skill', 'input', 'expected'),
                           'trace': ('skill', 'input'), 'inspect': ('skill',)}.items():
        p = sub.add_parser(command)
        for arg in names:
            p.add_argument(arg)
    sub.add_parser('list')
    sub.add_parser('demo')
    args = parser.parse_args(argv)
    path = Path(args.memory)
    if args.command == 'demo':
        memory = new_memory()
        events = [teach(memory, 'letters', 'APPLE', 'FIVE'),
                  teach(memory, 'letters', 'BANANA', 'SIX'),
                  prove(memory, 'letters', 'CAT', 'THREE'),
                  trace(memory, 'letters', 'APPLE')]
        for name, event in zip(['teach APPLE', 'teach BANANA', 'test CAT', 'trace APPLE'], events):
            print(f'{name}: {json.dumps(event, ensure_ascii=False)}')
        return 0
    try:
        memory = read_memory(path)
        if args.command == 'list':
            print(json.dumps({name: {'examples': len(s['examples']), 'rules': matching_rules(s['examples'])}
                              for name, s in memory['skills'].items()}, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'inspect':
            record = memory['skills'].get(args.skill)
            result = {'status': 'NOT_FOUND'} if record is None else {
                'status': 'FOUND', 'examples': record['examples'],
                'candidate_rules': matching_rules(record['examples']),
                'tests': record['tests']}
        elif args.command in ('teach', 'correct'):
            result = teach(memory, args.skill, args.input, args.output, correct=args.command == 'correct')
            save_memory(path, memory)
        elif args.command == 'run':
            result = predict(memory, args.skill, args.input)
        elif args.command == 'prove':
            result = prove(memory, args.skill, args.input, args.expected)
            save_memory(path, memory)
        elif args.command == 'trace':
            result = trace(memory, args.skill, args.input)
        else:
            raise AssertionError('command parser unexpected')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({'status': 'INSPECTION_NEEDED', 'message': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())