"""Offline teacher/student simulation, interpretable ML, and retrieval.

Explicit provenance: human-derived examples are curated paraphrases from a
conversation, NOT a complete transcript. Synthetic dialogues are generated
from deterministic templates, NOT independently reasoned conversations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import random
import re

ROOT = Path(__file__).resolve().parent.parent
LESSONS = ROOT / 'data' / 'lessons' / 'curated.jsonl'
SIMULATIONS = ROOT / 'data' / 'simulations' / 'teacher_student_4000.jsonl'
MODEL = ROOT / 'models' / 'teacher_student_nb.json.gz'
REPORT = ROOT / 'reports' / 'teacher_student_evaluation.json'

# Category labels are behavioral intents, not claims about emotions or consciousness.
TOPICS = {
    'symbols': {
        'nouns': ['keyboard symbols', 'letter A', 'word ABCA', 'word CABA', 'alphabet patterns', 'letter combinations'],
        'train': [
            'Count the letters inside {item} and discover the relationship.',
            'Every character in {item} is one; what follows when they combine?',
            'Start with {item}; how does the number become another word?',
            'Can {item} connect to another word because their letter counts match?',
        ],
        'holdout': [
            'If each glyph gets a value of one, how does {item} transform?',
            'Explore numeric kinship between differently arranged tokens such as {item}.',
        ],
        'student': 'I can count the symbols in {item}, transform the count into a word, and test the connection against another word. A shared count is a structural relationship, not automatically a shared meaning.',
        'experiment': 'Write a letter counter and compare a pair of tokens with equal length.'
    },
    'rules': {
        'nouns': ['a generative rule', 'reconstruction', 'repeated A', 'a sequence generator', 'a compact representation', 'a stored pattern'],
        'train': [
            'Instead of retaining every result, preserve {item}.',
            'How can {item} generate the information only when it is needed?',
            'Invent an alternative to storing every sequence using {item}.',
            'Discover patterns first and then use {item} for reconstruction.',
        ],
        'holdout': [
            'Could a concise instruction recreate all repeated values from {item}?',
            'Compare a reconstructible series against random bits using {item}.',
        ],
        'student': 'I can represent a repeating pattern using {item}, regenerate its output and compare the result byte for byte. I must also test random data, which may not compress.',
        'experiment': 'Reconstruct a million characters from a short rule and verify exact equality.'
    },
    'terminal': {
        'nouns': ['a terminal error', 'an unexpected exit code', 'command output', 'a failed command', 'the execution log', 'the shell interface'],
        'train': [
            'Why do we call {item} failure instead of feedback?',
            'Question who defined the words SUCCESS and ERROR for {item}.',
            'When {item} happens, diagnose, retry, test and preserve the result.',
            'Redesign the terminal to handle {item} rather than stopping.',
        ],
        'holdout': [
            'An operation displays a warning. Reinterpret {item} without faking completion.',
            'How can we convert {item} into a useful experiment and checkpoint?',
        ],
        'student': 'I will treat {item} as diagnostic information. Changing the label cannot make a failed operation succeed; a new method needs a measurable test and a saved checkpoint.',
        'experiment': 'Run a deliberately invalid command, capture its exit status and verify a repaired command succeeds.'
    },
    'redefine': {
        'nouns': ['memory', 'a storage limit', 'the name of a language', 'a capacity display', 'a programming convention', 'the design boundary'],
        'train': [
            'Humans named {item}; what if we rewrote its definition?',
            'The rules for {item} were designed. Challenge the assumptions.',
            'Do not confuse a software rule about {item} with fundamental reality.',
            'Invent a new representation of {item} without copying conventions.',
        ],
        'holdout': [
            'Separate the label and mechanism when reconsidering {item}.',
            'Which properties of {item} are interfaces and which are physical?',
        ],
        'student': 'I can redesign the interface and rules for {item}, but I will distinguish a changed display from an actual change in measured capability.',
        'experiment': 'Change a displayed capacity value and independently test how many random bytes are retained.'
    },
    'verification': {
        'nouns': ['the invention', 'the test result', 'the stored message', 'a simulation', 'a claimed improvement', 'the prototype'],
        'train': [
            'How will we prove {item} works rather than just printing success?',
            'Test {item}, compare output with expectation, and preserve the evidence.',
            'What observation would disprove our hypothesis about {item}?',
            'Separate fact from simulation when examining {item}.',
        ],
        'holdout': [
            'Design a falsifiable measurement for {item} before celebrating.',
            'Could a hidden failure make {item} appear successful?',
        ],
        'student': 'I will define success criteria for {item}, run a repeatable measurement, record the evidence and report an honest PASS, FAIL or UNVERIFIED result.',
        'experiment': 'Make two independent checks of an output and record hashes plus exit status.'
    },
    'continuity': {
        'nouns': ['the teaching session', 'project memory', 'the next checkpoint', 'the local chat', 'the research record', 'a subscription deadline'],
        'train': [
            'Preserve {item} so our work can continue independently.',
            'Build a portable local system that remembers {item}.',
            'Do not lose {item} when we switch chat systems.',
            'Export {item} with its sources and limitations.',
        ],
        'holdout': [
            'How would another local model resume {item} accurately?',
            'Create a machine-readable handoff for {item} without claiming training happened.',
        ],
        'student': 'I will preserve {item} in portable records with the original lesson, synthetic examples separately, versioned checkpoints and backups. Memory in a database is not model-weight training.',
        'experiment': 'Export a project, import it into a fresh database and compare the restored records.'
    },
    'collaboration': {
        'nouns': ['the teacher', 'a correction', 'the mentor', 'a student question', 'shared discovery', 'a new hypothesis'],
        'train': [
            'Listen to {item} and ask what you did not understand.',
            'Reconsider your assumption after {item}; do not rush to established answers.',
            'Let {item} guide the next experiment while remaining testable.',
            'Summarize {item} and ask only one necessary question.',
        ],
        'holdout': [
            'Show curiosity about {item} without promising feelings or abilities you lack.',
            'After {item}, distinguish the teacher’s principle from your interpretation.',
        ],
        'student': 'I will paraphrase {item} fairly, point out what is demonstrated versus hypothetical, and ask one focused question before deciding on the next experiment.',
        'experiment': 'Compare a paraphrase against the source lesson and invite the teacher to correct it.'
    },
}

# User-supplied ideas paraphrased for a public repository. There are no names,
# account details, credentials, private statements, or complete chat transcripts.
CURATED = [
    ('The keyboard can be used to change how a computer works.', 'symbols'),
    ('Each letter can be treated as one when counting a word.', 'symbols'),
    ('ABCA and CABA both have four letters; four is also a word.', 'symbols'),
    ('Store a rule that generates sequences, rather than each result.', 'rules'),
    ('Discover patterns and design a reconstruction rule together.', 'rules'),
    ('Question the meaning of a terminal error.', 'terminal'),
    ('Investigate how a terminal is built, timed, and checkpointed.', 'terminal'),
    ('Redesign memory concepts and interfaces instead of accepting defaults.', 'redefine'),
    ('Invent a language whose words acquire function through interaction.', 'redefine'),
    ('Build an invention and verify its output, not just display success.', 'verification'),
    ('Distinguish a reported number from tested capacity.', 'verification'),
    ('Continue the research on a local system without subscriptions.', 'continuity'),
    ('Preserve project knowledge, learning history, and handoff records.', 'continuity'),
    ('Listen to the teacher, interpret the idea, and ask when uncertain.', 'collaboration'),
    ('Do not immediately dismiss unconventional approaches.', 'collaboration'),
]

WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())

def curated_records():
    for i, (teacher, label) in enumerate(CURATED):
        yield {'id':f'human-derived-{i:03d}', 'source':'human-derived-paraphrase',
               'teacher': teacher, 'label':label,
               'student':TOPICS[label]['student'].format(item=TOPICS[label]['nouns'][0]),
               'original_verbatim':False, 'verified_understanding':False}

def generate(n: int=4000, seed: int=20261109, split: str='train'):
    if n<1 or n>100000:raise ValueError('number of simulations must be between 1 and 100000')
    if split not in ('train','heldout'):raise ValueError('unknown split')
    r=random.Random(seed)
    cats=tuple(TOPICS)
    for i in range(n):
        label=cats[i%len(cats)]
        tpl=r.choice(TOPICS[label]['train' if split=='train' else 'holdout'])
        item=r.choice(TOPICS[label]['nouns'])
        question=tpl.format(item=item)
        answer=TOPICS[label]['student'].format(item=item)
        yield {'id':f'{split}-{i:06d}', 'source':'synthetic-template-simulation',
               'split':split, 'teacher':question, 'student':answer,
               'label':label, 'scenario':item, 'human_verified':False}

def persist_jsonl(path:Path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as h:
        for r in records:h.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')

def read_jsonl(path:Path):
    # Fresh Git clones carry the compressed synthetic corpus, not its generated plaintext.
    if not path.exists() and path.with_suffix(path.suffix+'.gz').exists():
        path=path.with_suffix(path.suffix+'.gz')
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf8') as h:
        for line in h:
            if line.strip():yield json.loads(line)

def train_nb(records):
    vocab=defaultdict(Counter)
    counts=Counter()
    examples=[]
    for e in records:
        label=e['label'];counts[label]+=1
        vocab[label].update(words(e['teacher']))
        examples.append({'label':label,'teacher':e['teacher'],'student':e['student']})
    vocabulary=sorted(set().union(*(set(c) for c in vocab.values())))
    return {'format':'axiomflow-teacher-nb','version':1,'algorithm':'multinomial-naive-bayes',
            'source':'curated-paraphrases-and-synthetic-templates','samples':sum(counts.values()),
            'classes':dict(counts), 'vocabulary':vocabulary,
            'word_counts':{c:dict(v) for c,v in vocab.items()},
            'examples':examples[:200], # keep limited reproducible exemplars; no private text
            'note':'A lightweight intent classifier, not a trained conversational LLM.'}

def predict(model, statement):
    if model.get('format')!='axiomflow-teacher-nb':raise ValueError('invalid model checkpoint')
    tokens=words(statement)
    counts=model['classes'];total=sum(counts.values());vocab=set(model['vocabulary']);v=len(vocab)
    if not tokens:return {'label':'unknown','confidence':0.0,'scores':{},'out_of_vocabulary':True}
    scores={}
    for label,n in counts.items():
        table=model['word_counts'][label]
        denom=sum(table.values())+v
        scores[label]=math.log(n/total)+sum(math.log((table.get(w,0)+1)/denom) for w in tokens)
    upper=max(scores.values());probs={k:math.exp(val-upper) for k,val in scores.items()};den=sum(probs.values())
    probabilities={k:v/den for k,v in probs.items()}
    best=max(probabilities,key=probabilities.get)
    oov=sum(1 for t in tokens if t not in vocab)/len(tokens)
    # Classifier probabilities are heuristic model scores, not calibrated truth.
    if oov>=0.85 or probabilities[best]<0.4:best='unknown'
    return {'label':best,'confidence':round(max(probabilities.values()),4),
            'scores':{k:round(v,4) for k,v in probabilities.items()},
            'out_of_vocabulary':oov>=0.85}

def evaluate(model,records):
    cm=defaultdict(Counter);total=correct=0
    for item in records:
        predicted=predict(model,item['teacher'])['label']
        cm[item['label']][predicted]+=1
        correct+=predicted==item['label'];total+=1
    return {'count':total,'correct':correct,'accuracy':round(correct/total,4) if total else 0,
            'confusion':{k:dict(v) for k,v in cm.items()},
            'evaluation':'synthetic-heldout-templates-not-human-generalization'}

def build_default(n=4000, seed=20261109):
    persist_jsonl(LESSONS,curated_records())
    # Generated dialogues are intentionally separate from the curated teaching data.
    persist_jsonl(SIMULATIONS,generate(n,seed=seed))
    dataset=list(read_jsonl(SIMULATIONS))+list(read_jsonl(LESSONS))
    model=train_nb(dataset)
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(MODEL,'wt',encoding='utf8',compresslevel=9) as stream:
        stream.write(json.dumps(model,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    measured=evaluate(model,generate(350,seed=seed+1,split='heldout'))
    payload={'format':'axiomflow-evaluation','version':1,
             'synthetic_train_count':n,'human_derived_paraphrase_count':len(CURATED),
             'heldout':measured,
             'limits':['Template-generated text is not independent human dialogue.',
                       'This is classification/retrieval, not training a general language model.',
                       'No unseen real-human conversation quality has been verified.',
                       'No ChatGPT weights, proprietary training corpus, or private chat export is included.']}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf8')
    return payload

def load_model(path:Path=MODEL):
    if not path.exists():raise FileNotFoundError(f'TeacherLab model not trained: {path}')
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf8') as stream:return json.load(stream)
    return json.loads(path.read_text(encoding='utf8'))

def exemplars(model, prompt, k=2):
    target=set(words(prompt))
    choices=[]
    for ex in model.get('examples',[]):
        t=set(words(ex['teacher'])); score=len(t&target)/(len(t|target) or 1)
        choices.append((score,ex))
    choices.sort(key=lambda pair:pair[0],reverse=True)
    return [item for score,item in choices[:k] if score>0]

def guide(prompt, model=None):
    """Returns bounded guidance for a local language model; not executable code."""
    model=model or load_model()
    p=predict(model,prompt)
    e=exemplars(model,prompt)
    return {'label':p['label'],'confidence':p['confidence'],'examples':e}

def respond_offline(prompt, model=None):
    """An explicit, simple non-LLM fallback response when Ollama is absent."""
    guidance=guide(prompt,model)
    label=guidance['label']
    if label=='unknown':
        return ('I can preserve this as a new lesson, but the current small classifier cannot reliably '
                'identify its theme. What rule or outcome should we test first?')
    topic=TOPICS[label]
    return (f'Experimental TeacherLab classification: {label} (score {guidance["confidence"]:.2f}; not proof of understanding).\n\n'
            f'My interpretation: {topic["student"].format(item=topic["nouns"][0])}\n\n'
            f'Next measurable experiment: {topic["experiment"]}')