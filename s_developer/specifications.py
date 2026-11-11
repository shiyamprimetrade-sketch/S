"""Immutable project specifications in the existing S SQLite memory boundary.

Requirements are data, never executable instructions. Approval writes a scoped
S decision note; it does not grant filesystem, shell, network or model authority.
"""
import json
from .intent import parse_intent
from continuum.storage import new_id, now

QUESTIONS = {
 'purpose': 'What should this project accomplish?',
 'users': 'Who will use it?',
 'features': 'Which actions must users be able to perform? One per line.',
 'platform': 'Where must it run: browser, Windows, phone, or another platform?',
 'data': 'What data must it store or integrate? Enter none if no data is stored.',
 'constraints': 'What privacy, accessibility, budget, offline or performance constraints apply? Enter none if none.',
 'acceptance': 'What observable checks would convince you it works? One per line.',
}
SCHEMA = '''
CREATE TABLE IF NOT EXISTS developer_specs(
 project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 revision INTEGER NOT NULL,
 document TEXT NOT NULL,
 created_at TEXT NOT NULL,
 PRIMARY KEY(project_id, revision)
);
'''


def text(value, label, limit):
    if not isinstance(value, str) or len(value.strip()) > limit:
        raise ValueError(f'{label} must be text up to {limit} characters')
    if '\x00' in value:
        raise ValueError('Null characters are not supported')
    return value.strip()


def structure(request, answers):
    intent = parse_intent(request)
    if not isinstance(answers, dict) or set(answers) - set(QUESTIONS):
        raise ValueError('Unknown requirements fields')
    fields = {key: text(answers.get(key, ''), key, 5000) for key in QUESTIONS}
    missing = [dict(field=key, question=question) for key, question in QUESTIONS.items() if not fields[key]]
    lines = lambda key: [v.strip() for v in fields[key].splitlines() if v.strip()]
    # This proposal is deliberately conservative, never a claimed model decision.
    stack = {'website': ['HTML', 'CSS', 'JavaScript'], 'api': ['Python'],
             'automation': ['Python']}.get(intent['kind'], [])
    return {'schema_version': 1, 'intent': intent, 'answers': fields,
            'functional_requirements': lines('features'),
            'nonfunctional_requirements': lines('constraints'),
            'acceptance_criteria': lines('acceptance'), 'clarifications': missing,
            'technology_proposal': {'languages': stack, 'status': 'proposal-only' if stack else 'needs-review',
                                    'reason': 'Conservative starting point; platform and integration review still required'},
            'plan': ['Resolve missing requirements', 'Review and approve specification',
                     'Review architecture and toolchain availability', 'Implement in a bounded project workspace',
                     'Build and test against acceptance criteria', 'Deliver with verification evidence'],
            'generation': {'provider': None, 'status': 'not-run',
                           'limitation': 'No reliable autonomous code-generation capability established for the native model'},
            'permissions': {'shell': False, 'network': False, 'installation': False,
                            'project_file_execution': False}}


class Specifications:
    def __init__(self, store, engine_config=None):
        self.store = store
        with store.connect() as db:
            db.executescript(SCHEMA)
        from .engine import DevelopmentEngine
        self.engine=DevelopmentEngine(self,engine_config)

    def latest(self, project_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            row = db.execute('SELECT document FROM developer_specs WHERE project_id=? ORDER BY revision DESC LIMIT 1',
                             (project_id,)).fetchone()
        if row is None:
            raise ValueError('No saved developer specification for this project')
        return json.loads(row[0])

    def history(self, project_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT document FROM developer_specs WHERE project_id=? ORDER BY revision DESC', (project_id,))]

    def projects(self):
        with self.store.connect() as db:
            return [dict(r) for r in db.execute('''SELECT p.id,p.name,MAX(d.revision) AS revision
                FROM projects p JOIN developer_specs d ON d.project_id=p.id GROUP BY p.id ORDER BY p.created_at,p.name''')]

    def save(self, body):
        if not isinstance(body, dict) or set(body) - {'name','request','answers','project_id','expected_revision','approve'}:
            raise ValueError('Invalid specification request')
        spec = structure(body.get('request'), body.get('answers', {}))
        name = text(body.get('name', ''), 'Project name', 100)
        if not name:
            raise ValueError('Project name is required')
        approve = body.get('approve', False)
        if type(approve) is not bool:
            raise ValueError('Approval must be boolean')
        if approve and spec['clarifications']:
            raise ValueError('Answer all clarification questions before approval')
        pid = body.get('project_id')
        expected = body.get('expected_revision', 0)
        if type(expected) is not int or expected < 0:
            raise ValueError('Invalid expected revision')
        if pid is not None and not isinstance(pid, str):
            raise ValueError('Invalid project id')
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if pid is None:
                if expected != 0:
                    raise ValueError('New projects must start at revision zero')
                pid = new_id()
                db.execute('INSERT INTO projects VALUES (?,?,?)', (pid, name, now()))
            else:
                project = db.execute('SELECT id FROM projects WHERE id=?', (pid,)).fetchone()
                if project is None:
                    raise ValueError('Unknown project')
            current = db.execute('SELECT COALESCE(MAX(revision),0) FROM developer_specs WHERE project_id=?', (pid,)).fetchone()[0]
            if expected != current:
                raise ValueError('Specification changed; reopen the project before saving')
            spec.update(project_id=pid, name=name, revision=current+1, created_at=now(),
                        status='approved' if approve else 'draft')
            document = json.dumps(spec, ensure_ascii=False)
            db.execute('UPDATE projects SET name=? WHERE id=?', (name, pid))
            db.execute('INSERT INTO developer_specs VALUES (?,?,?,?)', (pid, current+1, document, spec['created_at']))
            if approve:
                db.execute('INSERT INTO notes VALUES (?,?,?,?,?,?,?)',
                           (new_id(), pid, f'Developer specification revision {current+1}', document,
                            'decision', 'Explicit local user approval; requirements only', now()))
            db.execute('INSERT INTO events(project_id,action,detail,created_at) VALUES (?,?,?,?)',
                       (pid, 'developer.specification', json.dumps({'revision':current+1, 'status':spec['status']}), now()))
        return spec

    def rollback(self, body):
        if not isinstance(body, dict) or set(body) != {'project_id','revision','expected_revision','confirm'}:
            raise ValueError('Invalid rollback request')
        if body['confirm'] is not True or type(body['revision']) is not int:
            raise PermissionError('Explicit rollback confirmation required')
        old = next((s for s in self.history(body['project_id']) if s['revision'] == body['revision']), None)
        if old is None:
            raise ValueError('Unknown checkpoint')
        # Restored content is a new draft: an old approval never grants new authority.
        return self.save({'project_id': body['project_id'], 'name': old['name'],
                          'request': old['intent']['request'], 'answers': old['answers'],
                          'expected_revision': body['expected_revision'], 'approve': False})

    def request(self, verb, path, query, body):
        if verb == 'GET':
            pid = query.get('project_id', [''])[0]
            if path == '/api/developer/questions': return QUESTIONS
            if path == '/api/developer/projects': return self.projects()
            if path == '/api/developer/spec': return self.latest(pid)
            if path == '/api/developer/history': return self.history(pid)
            if path == '/api/developer/toolchains':
                from .registry import catalog
                return catalog()
        elif verb == 'POST':
            if path == '/api/developer/intent':
                if not isinstance(body, dict): raise ValueError('Expected object')
                return parse_intent(body.get('request'))
            if path == '/api/developer/spec': return self.save(body)
            if path == '/api/developer/rollback': return self.rollback(body)
        return self.engine.request(verb,path,query,body)
