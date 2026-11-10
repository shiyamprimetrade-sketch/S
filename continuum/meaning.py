"""S v0.6 — human-reviewed teaching memory, concept aliases and provenance.

This is deterministic, local retrieval and explicit teaching, NOT pretrained
semantic understanding or automatic neural-network fine-tuning. Archived chats
and negative feedback are suggestions until the teacher approves them.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from .storage import Store, new_id, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS s_concepts (
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 label TEXT NOT NULL, variants_json TEXT NOT NULL, created_at TEXT NOT NULL,
 UNIQUE(project_id,label)
);
CREATE INDEX IF NOT EXISTS idx_s_concepts_project ON s_concepts(project_id);
CREATE TABLE IF NOT EXISTS s_lessons (
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 title TEXT NOT NULL, principle TEXT NOT NULL, cues_json TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('candidate','approved','rejected','superseded')),
 origin TEXT NOT NULL CHECK(origin IN ('teacher','archive','feedback','import')),
 source_id TEXT, supersedes_id TEXT, version INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_s_lessons_project ON s_lessons(project_id,status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_s_lessons_source ON s_lessons(project_id,origin,source_id)
 WHERE source_id IS NOT NULL;
"""
STOP = {'i','you','we','they','my','me','our','your','this','that','the','and','for',
        'from','have','has','had','with','will','would','could','should','make','want',
        'can','are','were','was','what','when','where','why','which','how','a','an',
        'of','to','in','on','at','it','is','be','do','does','not','new','please',
        'explain','tell','think','about','something','everything','more','things'}
WORD = re.compile(r"[^\W_]+", re.UNICODE)
CORRECTIONS = ('no,', 'no ', 'incorrect', 'that is wrong', "that's wrong",
               'you misunderstood', 'you are wrong', 'i meant', 'actually,',
               'the reason i', 'not accurate', 'not what i', 'you assumed')


def _text(value, limit, name):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be text')
    value = value.strip()
    if not 1 <= len(value) <= limit:
        raise ValueError(f'{name} must be 1–{limit} characters')
    return value


def _strings(values, *, maximum=12, maxlen=180, min_count=1):
    if not isinstance(values, list) or not min_count <= len(values) <= maximum:
        raise ValueError(f'Expected {min_count}–{maximum} phrases')
    result = []
    for value in values:
        v = _text(value, maxlen, 'Phrase')
        if v.casefold() not in {x.casefold() for x in result}:
            result.append(v)
    if len(result) < min_count:
        raise ValueError('Not enough distinct phrases')
    return result


def _words(text):
    return [w for w in WORD.findall(text.casefold()) if len(w) > 1 and w not in STOP]


def _record(row):
    r = dict(row)
    if 'cues_json' in r:
        r['cues'] = json.loads(r.pop('cues_json'))
    if 'variants_json' in r:
        r['variants'] = json.loads(r.pop('variants_json'))
    return r


class TeacherMemory:
    def __init__(self, store: Store):
        self.store = store
        with store.connect() as db:
            db.executescript(SCHEMA)

    def concepts(self, project_id):
        self.store.get_project(project_id)
        with self.store.connect() as db:
            return [_record(x) for x in db.execute(
                'SELECT * FROM s_concepts WHERE project_id=? ORDER BY label', (project_id,))]

    def define_concept(self, project_id, label, variants):
        self.store.get_project(project_id)
        label = _text(label, 65, 'Concept label').casefold()
        if not re.fullmatch(r'[\w -]+', label, re.UNICODE):
            raise ValueError('Invalid concept label')
        variants = _strings(variants, maximum=15, maxlen=75, min_count=2)
        new_phrases = {tuple(_words(v)) for v in variants}
        if () in new_phrases:
            raise ValueError('Every variant needs a meaningful word')
        for concept in self.concepts(project_id):
            if concept['label'] != label and new_phrases & {tuple(_words(v)) for v in concept['variants']}:
                raise ValueError('A phrase cannot belong to competing concepts in the same project')
        item = dict(id=new_id(), project_id=project_id, label=label,
                    variants_json=json.dumps(variants, ensure_ascii=False), created_at=now())
        with self.store.connect() as db:
            if db.execute('SELECT 1 FROM s_concepts WHERE project_id=? AND label=?',
                          (project_id, label)).fetchone():
                db.execute('UPDATE s_concepts SET variants_json=? WHERE project_id=? AND label=?',
                           (item['variants_json'], project_id, label))
            else:
                db.execute('INSERT INTO s_concepts VALUES (:id,:project_id,:label,:variants_json,:created_at)',item)
        return dict(label=label, variants=variants, state='teacher_defined_equivalence')

    def _validate_source(self, project_id, source_id, origin):
        if source_id is None:
            return None
        if not isinstance(source_id, str) or len(source_id) > 100:
            raise ValueError('Invalid source identifier')
        with self.store.connect() as db:
            if origin == 'archive':
                row = db.execute('SELECT m.role,m.content FROM messages m JOIN sessions s ON s.id=m.session_id '
                                 'WHERE m.id=? AND s.project_id=?', (source_id, project_id)).fetchone()
                if row is None or row['role'] != 'user':
                    raise ValueError('Archived teacher message not found in this project')
            if origin == 'feedback':
                row = db.execute('SELECT f.rating FROM feedback f JOIN messages m ON m.id=f.message_id '
                                 'JOIN sessions s ON s.id=m.session_id WHERE f.message_id=? AND s.project_id=?',
                                 (source_id, project_id)).fetchone()
                if row is None or row['rating'] != -1:
                    raise ValueError('Negative feedback not found in this project')
        return source_id

    def teach(self, project_id, title, principle, cues, *, status='approved',
              origin='teacher', source_id=None):
        self.store.get_project(project_id)
        title = _text(title, 140, 'Lesson title')
        principle = _text(principle, 3000, 'Lesson principle')
        cues = _strings(cues, maximum=12, maxlen=220)
        if status not in ('candidate', 'approved'):
            raise ValueError('Lesson creation must be candidate or approved')
        if origin not in ('teacher', 'archive', 'feedback', 'import'):
            raise ValueError('Unknown lesson origin')
        self._validate_source(project_id, source_id, origin)
        item = dict(id=new_id(), project_id=project_id, title=title, principle=principle,
                    cues_json=json.dumps(cues, ensure_ascii=False), status=status,
                    origin=origin, source_id=source_id, supersedes_id=None, version=1,
                    created_at=now(), updated_at=now())
        with self.store.connect() as db:
            db.execute('INSERT INTO s_lessons VALUES (:id,:project_id,:title,:principle,:cues_json,'
                       ':status,:origin,:source_id,:supersedes_id,:version,:created_at,:updated_at)', item)
        self.store.log(project_id, 'teacher-memory', f"lesson={item['id']} status={status} origin={origin}")
        return _record(item)

    def lessons(self, project_id, status=None, limit=200):
        self.store.get_project(project_id)
        if status is not None and status not in ('candidate', 'approved', 'rejected', 'superseded'):
            raise ValueError('Invalid lesson status')
        with self.store.connect() as db:
            if status is None:
                rows = db.execute('SELECT * FROM s_lessons WHERE project_id=? ORDER BY rowid DESC LIMIT ?',
                                  (project_id, min(1000, max(1, int(limit)))))
            else:
                rows = db.execute('SELECT * FROM s_lessons WHERE project_id=? AND status=? '
                                  'ORDER BY rowid DESC LIMIT ?',
                                  (project_id, status, min(1000, max(1, int(limit)))))
            return [_record(x) for x in rows]

    def review(self, project_id, lesson_id, decision):
        self.store.get_project(project_id)
        if decision not in ('approved', 'rejected'):
            raise ValueError('Expected approval or rejection')
        with self.store.connect() as db:
            row = db.execute('SELECT status FROM s_lessons WHERE id=? AND project_id=?',
                             (lesson_id, project_id)).fetchone()
            if not row or row['status'] != 'candidate':
                raise ValueError('Unknown or already reviewed candidate')
            db.execute('UPDATE s_lessons SET status=?,updated_at=? WHERE id=? AND project_id=?',
                       (decision, now(), lesson_id, project_id))
        self.store.log(project_id, 'teacher-review', f'lesson={lesson_id} decision={decision}')
        return dict(id=lesson_id, status=decision)

    def revise(self, project_id, lesson_id, principle, cues):
        self.store.get_project(project_id)
        principle = _text(principle, 3000, 'Corrected principle')
        cues = _strings(cues, maximum=12, maxlen=220)
        with self.store.connect() as db:
            row = db.execute('SELECT * FROM s_lessons WHERE id=? AND project_id=?',
                             (lesson_id, project_id)).fetchone()
            if row is None or row['status'] != 'approved':
                raise ValueError('Only an approved lesson can be revised')
            item = dict(id=new_id(), project_id=project_id, title=row['title'], principle=principle,
                        cues_json=json.dumps(cues, ensure_ascii=False), status='approved',
                        origin='teacher', source_id=None, supersedes_id=lesson_id,
                        version=row['version'] + 1, created_at=now(), updated_at=now())
            db.execute("UPDATE s_lessons SET status='superseded',updated_at=? WHERE id=?",(now(),lesson_id))
            db.execute('INSERT INTO s_lessons VALUES (:id,:project_id,:title,:principle,:cues_json,'
                       ':status,:origin,:source_id,:supersedes_id,:version,:created_at,:updated_at)',item)
        self.store.log(project_id, 'teacher-revision', f'previous={lesson_id} revised={item["id"]}')
        return _record(item)

    def suggest_archive_corrections(self, project_id, limit=30):
        """Only within selected project; imported statements stay UNAPPROVED.

        Uses explicit correction markers, not arbitrary messages or assistant text.
        The heuristic may miss lessons or flag unrelated statements.
        """
        self.store.get_project(project_id)
        count = 0
        with self.store.connect() as db:
            sessions = [dict(x) for x in db.execute(
                'SELECT id,title FROM sessions WHERE project_id=? ORDER BY rowid DESC LIMIT 300',
                (project_id,))]
        for session in sessions:
            with self.store.connect() as db:
                rows = [dict(x) for x in db.execute(
                    'SELECT id,role,content FROM messages WHERE session_id=? ORDER BY rowid',
                    (session['id'],))]
            previous_assistant = False
            last_user = ''
            for message in rows:
                if message['role'] == 'assistant':
                    previous_assistant = True
                    continue
                content = message['content'].strip()
                lower = content.casefold()
                if previous_assistant and any(lower.startswith(x) for x in CORRECTIONS) and len(content) >= 12:
                    cues = [last_user[:220] if last_user else session['title'][:120]]
                    try:
                        self.teach(project_id, 'Possible correction — '+session['title'][:95],
                                   content[:3000], cues, status='candidate',origin='archive',
                                   source_id=message['id'])
                        count += 1
                    except Exception as ex:
                        # The ONLY expected conflict is an already suggested source.
                        with self.store.connect() as db:
                            existing = db.execute('SELECT id FROM s_lessons WHERE project_id=? AND '
                                                  'origin=? AND source_id=?',
                                                  (project_id,'archive',message['id'])).fetchone()
                        if not existing:
                            raise ex
                last_user = content
                previous_assistant = False
                if count >= min(100, max(1,int(limit))):
                    return {'candidates_added':count, 'status':'unreviewed_suggestions'}
        return {'candidates_added':count,'status':'unreviewed_suggestions'}

    def propose_feedback(self, project_id, message_id, correction):
        """Negative feedback is tentative until the teacher reviews the candidate."""
        self._validate_source(project_id, message_id, 'feedback')
        with self.store.connect() as db:
            row = db.execute('SELECT m.session_id,m.rowid FROM messages m JOIN sessions s '
                             'ON s.id=m.session_id WHERE m.id=? AND s.project_id=?',
                             (message_id, project_id)).fetchone()
            preceding = db.execute('SELECT content FROM messages WHERE session_id=? AND role=? '
                                   'AND rowid<? ORDER BY rowid DESC LIMIT 1',
                                   (row['session_id'],'user',row['rowid'])).fetchone()
        cue = preceding['content'][:220] if preceding else 'Assistant correction'
        return self.teach(project_id,'Feedback — correction suggested',correction,[cue],
                          status='candidate',origin='feedback',source_id=message_id)

    def recall(self, project_id, question, limit=3, *, use_concepts=True):
        """Traceable retrieval with teacher-defined synonym equivalence; no hallucinated matches."""
        self.store.get_project(project_id)
        question = _text(question, 50000, 'Query')
        concepts = self.concepts(project_id) if use_concepts else []
        aliases = {}
        for concept in concepts:
            for variant in concept['variants']:
                ws = tuple(_words(variant))
                if ws:
                    aliases[ws] = '@' + concept['label'].replace(' ','_')
        widths = sorted({len(phrase) for phrase in aliases},reverse=True)
        def normalize(s):
            words = _words(s)
            output = []
            i = 0
            while i < len(words):
                matched = False
                for width in widths:
                    phrase = tuple(words[i:i+width])
                    if len(phrase) == width and phrase in aliases:
                        output.append(aliases[phrase]); i += width; matched = True; break
                if not matched:
                    output.append(words[i]); i += 1
            return set(output)
        query = normalize(question)
        scored = []
        for lesson in self.lessons(project_id,status='approved',limit=1000):
            cues = set().union(*(normalize(c) for c in lesson['cues']))
            titles = normalize(lesson['title'])
            principle = normalize(lesson['principle'])
            signals = query & (cues | titles | principle)
            score = 5*len(query & cues)+3*len(query & titles)+len(query & principle)
            if score <= 0:continue
            scored.append((score, lesson['version'], lesson['id'], lesson, signals))
        scored.sort(key=lambda x:(-x[0],-x[1],x[2]))
        return [{**lesson, 'score':score,'matched_terms':sorted(signals),
                 'explanation':'teacher-approved retrieval via exact or teacher-defined concept matches',
                 'is_verified_fact':False} for score,_,__,lesson,signals in scored[:max(1,min(10,int(limit)))]]

    def transfer(self, source_project_id, destination_project_id, lesson_id, include_concepts=False):
        """Explicit teacher-authorized transfer between otherwise isolated projects.

        Only approved lessons may move. The destination stores provenance
        as source-project:source-lesson, not a claim of scientific verification.
        """
        self.store.get_project(source_project_id)
        self.store.get_project(destination_project_id)
        if source_project_id == destination_project_id:
            raise ValueError('Choose a different destination project')
        with self.store.connect() as db:
            row = db.execute('SELECT * FROM s_lessons WHERE id=? AND project_id=? AND status=?',
                             (lesson_id,source_project_id,'approved')).fetchone()
        if not row:
            raise ValueError('Only teacher-approved lessons can be transferred')
        source = _record(row)
        provenance = source_project_id+':'+lesson_id
        with self.store.connect() as db:
            duplicate = db.execute('SELECT id FROM s_lessons WHERE project_id=? AND origin=? AND source_id=?',
                                   (destination_project_id,'import',provenance)).fetchone()
        if duplicate:
            raise ValueError('This lesson has already been transferred')
        copied = self.teach(destination_project_id,source['title'],source['principle'],source['cues'],
                            origin='import',source_id=provenance)
        copied_concepts = 0
        conflicts = []
        if include_concepts:
            for concept in self.concepts(source_project_id):
                try:
                    self.define_concept(destination_project_id,concept['label'],concept['variants'])
                    copied_concepts += 1
                except ValueError as error:
                    conflicts.append({'label':concept['label'],'error':str(error)})
        self.store.log(destination_project_id,'explicit-lesson-transfer',
                       f'from={source_project_id} lesson={lesson_id} new={copied["id"]}')
        return {'lesson':copied,'concepts_copied':copied_concepts,'conflicts':conflicts,
                'notice':'Teacher-authorized local transfer; still not independent verification'}

    def export(self, project_id):
        return {'lessons':self.lessons(project_id,limit=1000),
                'concepts':self.concepts(project_id)}

    def restore(self, project_id, document, message_id_map=None):
        """Import portable lesson history into an already new, isolated project."""
        self.store.get_project(project_id)
        if not isinstance(document, dict):raise ValueError('Bad teacher memory export')
        lessons = document.get('lessons',[])
        concepts = document.get('concepts',[])
        if not isinstance(lessons,list) or not isinstance(concepts,list):raise ValueError('Invalid teacher memory lists')
        if len(lessons)>1000 or len(concepts)>500:raise ValueError('Memory export exceeds limits')
        for c in concepts:
            self.define_concept(project_id,c['label'],c['variants'])
        old_new = {}
        message_id_map = message_id_map or {}
        for lesson in reversed(lessons):
            if lesson['status'] not in ('approved','candidate','rejected','superseded'):
                raise ValueError('Bad lesson status')
            origin = lesson['origin']
            source_id = lesson.get('source_id')
            if origin in ('archive','feedback'):
                # Map same-project message provenance to the new imported session.
                # If the referenced message is absent, keep the lesson but do not
                # claim its original source link is still valid.
                source_id = message_id_map.get(source_id)
                if source_id is None:
                    origin = 'import'
            item=self.teach(project_id,lesson['title'],lesson['principle'],lesson['cues'],
                            status='approved' if lesson['status']=='approved' else 'candidate',
                            origin=origin, source_id=source_id)
            old_new[lesson['id']] = item['id']
            with self.store.connect() as db:
                db.execute('UPDATE s_lessons SET status=?,version=?,supersedes_id=?,created_at=?,updated_at=? '
                           'WHERE id=?', (lesson['status'],int(lesson.get('version',1)),
                           old_new.get(lesson.get('supersedes_id')),
                           lesson.get('created_at') or now(),lesson.get('updated_at') or now(),item['id']))
        return {'lesson_id_map':old_new,'imported_lessons':len(old_new),'imported_concepts':len(concepts),
                'notice':'Same-project source messages were re-linked when present; otherwise provenance is marked as imported'}
