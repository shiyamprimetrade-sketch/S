"""S v0.7 — teacher-curated blind retrieval benchmarks.

These benchmarks measure whether S retrieves teacher-approved lessons, not
whether its generated prose is correct, creative, or generally intelligent.
Evaluation questions remain outside the learner's lesson cues and model prompts.
"""
from __future__ import annotations

import hashlib
import json
from .meaning import TeacherMemory
from .storage import Store, new_id, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS apprenticeship_cases(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 question TEXT NOT NULL, expected_lesson_id TEXT,
 created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_apprentice_cases_project ON apprenticeship_cases(project_id);
CREATE TABLE IF NOT EXISTS apprenticeship_runs(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 created_at TEXT NOT NULL, dataset_sha256 TEXT NOT NULL,
 memory_sha256 TEXT NOT NULL, metrics_json TEXT NOT NULL, results_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_apprentice_runs_project ON apprenticeship_runs(project_id,created_at);
"""


def digest(value):
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_question(question):
    if not isinstance(question, str) or not 6 <= len(question.strip()) <= 1000:
        raise ValueError('An unseen test question must be 6–1000 characters')
    return question.strip()


class Apprenticeship:
    def __init__(self, store: Store):
        self.store = store
        self.memory = TeacherMemory(store)
        with store.connect() as db:
            db.executescript(SCHEMA)

    def cases(self, project_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            return [dict(x) for x in db.execute(
                'SELECT * FROM apprenticeship_cases WHERE project_id=? ORDER BY rowid ASC',
                (project_id,))]

    def add_case(self, project_id, question, expected_lesson_id=None):
        """Teacher provides expected lesson or null for a no-retrieval control.

        Never insert benchmark questions into lesson cues, messages, or training.
        """
        self.store.get_project(project_id)
        question = validate_question(question)
        if expected_lesson_id == '':
            expected_lesson_id = None
        if expected_lesson_id is not None:
            if not isinstance(expected_lesson_id, str):
                raise ValueError('Expected lesson must be an identifier')
            with self.store.connect() as db:
                match = db.execute('SELECT cues_json FROM s_lessons WHERE id=? AND project_id=? '
                                   "AND status='approved'", (expected_lesson_id,project_id)).fetchone()
            if match is None:
                raise ValueError('Expected lesson must be approved and in this project')
            if question.casefold() in {c.strip().casefold() for c in json.loads(match['cues_json'])}:
                raise ValueError('A blind test cannot repeat an exact teaching cue')
        with self.store.connect() as db:
            existing = db.execute('SELECT question FROM apprenticeship_cases WHERE project_id=?',
                                  (project_id,))
            if any(question.casefold() == row['question'].casefold() for row in existing):
                raise ValueError('Duplicate held-out question')
            item = dict(id=new_id(), project_id=project_id,question=question,
                        expected_lesson_id=expected_lesson_id,created_at=now())
            db.execute('INSERT INTO apprenticeship_cases VALUES '
                       '(:id,:project_id,:question,:expected_lesson_id,:created_at)',item)
        self.store.log(project_id,'apprenticeship-case',f"case={item['id']} no_match={expected_lesson_id is None}")
        return item

    def delete_case(self, project_id, case_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            cur = db.execute('DELETE FROM apprenticeship_cases WHERE project_id=? AND id=?',
                             (project_id,case_id))
            if cur.rowcount != 1:
                raise ValueError('Held-out test not found in this project')
        self.store.log(project_id,'apprenticeship-delete',f'case={case_id}')
        return {'deleted':True,'id':case_id}

    def readiness(self, project_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            sessions = db.execute('SELECT COUNT(*) FROM sessions WHERE project_id=?',(project_id,)).fetchone()[0]
            messages = db.execute('SELECT COUNT(*) FROM messages m JOIN sessions s ON s.id=m.session_id '
                                  'WHERE s.project_id=?',(project_id,)).fetchone()[0]
        approved = len(self.memory.lessons(project_id,'approved',1000))
        candidates = len(self.memory.lessons(project_id,'candidate',1000))
        return dict(project_id=project_id,conversations=sessions,messages=messages,
                    approved_lessons=approved,review_candidates=candidates,
                    held_out_questions=len(self.cases(project_id)),
                    actual_private_export_imported='not_verifiable_from_counts',
                    caveat='History counts do not prove a complete ChatGPT export was imported. '
                           'Benchmarks evaluate retrieval only, not model reasoning or training.')

    def runs(self, project_id, limit=10):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            rows=db.execute('SELECT * FROM apprenticeship_runs WHERE project_id=? '
                            'ORDER BY rowid DESC LIMIT ?', (project_id,min(30,max(1,int(limit)))))
            return [self._run_row(x) for x in rows]

    @staticmethod
    def _run_row(row):
        item=dict(row)
        item['metrics']=json.loads(item.pop('metrics_json'))
        item['results']=json.loads(item.pop('results_json'))
        return item

    def run(self, project_id):
        """Compare basic word matching to approved teacher-defined concept recall.

        Blind questions are NEVER used in the retrieval index. A stale expected
        lesson is reported and excluded from denominators rather than silently
        mapped to a different lesson.
        """
        cases = self.cases(project_id)
        if not cases:
            raise ValueError('Add at least one held-out question before running an evaluation')
        if len(cases)>200:
            raise ValueError('Evaluation limited to 200 held-out questions')
        lessons = self.memory.lessons(project_id,'approved',1000)
        approved = {x['id'] for x in lessons}
        snapshot = [{'id':x['id'],'version':x['version'],'title':x['title'],
                     'cues':x['cues'],'principle':x['principle']} for x in lessons]
        concepts = [{'label':c['label'],'variants':c['variants']}
                    for c in self.memory.concepts(project_id)]
        results=[]
        scores={'baseline':0,'taught_meaning':0}
        eligible=0
        stale=0
        for case in cases:
            expected=case['expected_lesson_id']
            if expected is not None and expected not in approved:
                results.append({'case_id':case['id'],'status':'stale_expected_lesson',
                                'expected_lesson_id':expected})
                stale += 1
                continue
            entry={'case_id':case['id'],'expected_lesson_id':expected,'status':'evaluated'}
            for label, use_concepts in [('baseline',False),('taught_meaning',True)]:
                hits=self.memory.recall(project_id,case['question'],limit=1,
                                        use_concepts=use_concepts)
                predicted=hits[0]['id'] if hits else None
                correct=predicted==expected
                scores[label] += int(correct)
                entry[label]={'prediction':predicted,'correct':correct,
                              'matched_terms':hits[0]['matched_terms'] if hits else []}
            eligible += 1
            results.append(entry)
        metrics={'total_cases':len(cases),'evaluated':eligible,'stale_excluded':stale,
                 'baseline_correct':scores['baseline'],
                 'taught_meaning_correct':scores['taught_meaning'],
                 'baseline_accuracy':scores['baseline']/eligible if eligible else None,
                 'taught_meaning_accuracy':scores['taught_meaning']/eligible if eligible else None,
                 'metric':'teacher-specified expected top-1 lesson, including no-match controls',
                 'claim':'Retrieval benchmark, not independent semantic understanding or generative accuracy'}
        record={'id':new_id(),'project_id':project_id,'created_at':now(),
                'dataset_sha256':digest([{'question':x['question'],'expected':x['expected_lesson_id']}
                                          for x in cases]),
                'memory_sha256':digest({'lessons':snapshot,'concepts':concepts}),
                'metrics_json':json.dumps(metrics,ensure_ascii=False,sort_keys=True),
                'results_json':json.dumps(results,ensure_ascii=False,sort_keys=True)}
        with self.store.connect() as db:
            db.execute('INSERT INTO apprenticeship_runs VALUES '
                       '(:id,:project_id,:created_at,:dataset_sha256,:memory_sha256,:metrics_json,:results_json)',record)
        self.store.log(project_id,'apprenticeship-run',
                       f"run={record['id']} measured={eligible} stale={stale} "
                       f"baseline={scores['baseline']} taught={scores['taught_meaning']}")
        return self._run_row(record)

    def export(self, project_id):
        return {'cases':self.cases(project_id),'runs':self.runs(project_id,limit=30)}

    def restore(self, project_id, document, lesson_id_map):
        """Restore old benchmarks to a new project without cross-project IDs.

        Historical results remain archival snapshots, not revalidated tests.
        """
        self.store.get_project(project_id)
        if not isinstance(document,dict) or not isinstance(document.get('cases',[]),list) \
           or not isinstance(document.get('runs',[]),list):
            raise ValueError('Invalid apprenticeship export')
        cases=document.get('cases',[]);runs=document.get('runs',[])
        if len(cases)>200 or len(runs)>30:raise ValueError('Apprenticeship export exceeds limits')
        mapped={}
        for case in cases:
            target=case.get('expected_lesson_id')
            if target is not None and target not in lesson_id_map:
                # A lesson deleted before export becomes an explicit stale expectation.
                mapped_target='unresolved:'+str(target)[:80]
            else:
                mapped_target=lesson_id_map.get(target)
            question=validate_question(case['question'])
            record={'id':new_id(),'project_id':project_id,'question':question,
                    'expected_lesson_id':mapped_target,'created_at':case.get('created_at') or now()}
            with self.store.connect() as db:
                db.execute('INSERT INTO apprenticeship_cases VALUES '
                           '(:id,:project_id,:question,:expected_lesson_id,:created_at)',record)
            mapped[case['id']]=record['id']
        for run in reversed(runs):
            if not isinstance(run.get('metrics'),dict) or not isinstance(run.get('results'),list):
                raise ValueError('Invalid historical benchmark')
            result=[]
            for row in run['results']:
                r=dict(row)
                r['case_id']=mapped.get(r.get('case_id'),'historical:unresolved')
                r['expected_lesson_id']=lesson_id_map.get(r.get('expected_lesson_id'),r.get('expected_lesson_id'))
                for mode in ('baseline','taught_meaning'):
                    if mode in r and isinstance(r[mode],dict):
                        r[mode]['prediction']=lesson_id_map.get(r[mode].get('prediction'),r[mode].get('prediction'))
                result.append(r)
            record={'id':new_id(),'project_id':project_id,'created_at':run.get('created_at') or now(),
                    'dataset_sha256':run['dataset_sha256'],'memory_sha256':run['memory_sha256'],
                    'metrics_json':json.dumps(run['metrics'],ensure_ascii=False),
                    'results_json':json.dumps(result,ensure_ascii=False)}
            with self.store.connect() as db:
                db.execute('INSERT INTO apprenticeship_runs VALUES '
                           '(:id,:project_id,:created_at,:dataset_sha256,:memory_sha256,:metrics_json,:results_json)',record)
        return {'restored_cases':len(cases),'restored_historical_runs':len(runs),
                'notice':'Historical hashes describe the source snapshot; rerun tests after importing.'}
