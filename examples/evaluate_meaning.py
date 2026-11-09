"""Small declared illustrative held-out test for teacher-defined concept retrieval.

This script is not a trained-LLM benchmark. The test authors chose the phrases.
It measures transfer between teacher-supplied paraphrases, with negatives.
"""
import json
import tempfile
from pathlib import Path
from continuum.meaning import TeacherMemory, _words
from continuum.storage import Store

LEARNING = [
    ('constraints','Investigate proposed technical limitations and compare alternatives.',
     ['What about limitations?','Which constraints must we measure?'],
     ['limitations','constraints','obstacles','barriers'],
     ['Can obstacles block our invention?', 'Do barriers require other designs?']),
    ('reasoning','Investigate assumptions, consider counterexamples and test outcomes.',
     ['Use rigorous analysis for this invention'],
     ['rigorous analysis','careful investigation','deep thinking'],
     ['Apply careful investigation now', 'Should we try deep thinking here?']),
    ('accuracy','Accuracy requires measurable reference conditions and verified results.',
     ['What makes a result accurate?'],
     ['accurate','correct','right result','valid result'],
     ['Is this result correct?', 'Does a right result need evidence?']),
]
NEGATIVES = ['Bake chocolate cookies', 'Where is the nearest garden?', 'Organize music playlists']


def run():
    with tempfile.TemporaryDirectory() as temp:
        store=Store(Path(temp)/'test.sqlite3')
        pid=store.create_project('Benchmark: no private user data')['id']
        memory=TeacherMemory(store)
        examples=[]
        for label,principle,cues,variants,heldouts in LEARNING:
            memory.teach(pid,label,principle,cues)
            memory.define_concept(pid,label,variants)
            examples.extend((q,label) for q in heldouts)
        examples.extend((q,None) for q in NEGATIVES)
        lessons=memory.lessons(pid,status='approved')
        def baseline(q):
            terms=set(_words(q))
            ranked=[]
            for lesson in lessons:
                ref=set(_words(' '.join(lesson['cues'])))
                score=len(terms & ref)
                if score:ranked.append((score,lesson['title']))
            ranked.sort(key=lambda t:-t[0])
            return ranked[0][1] if ranked else None
        records=[]
        for query,expected in examples:
            got=memory.recall(pid,query,limit=1)
            predicted=got[0]['title'] if got else None
            records.append({'query':query,'expected':expected,'keyword_baseline':baseline(query),
                            'teacher_memory':predicted})
        baseline_hits=sum(r['keyword_baseline']==r['expected'] for r in records)
        improved_hits=sum(r['teacher_memory']==r['expected'] for r in records)
        report={'scope':'Hand-authored 9-query demonstration, not general language understanding',
                'size':len(records),'baseline_correct':baseline_hits,
                'teacher_memory_correct':improved_hits,'records':records,
                'limitations':'Aliases and teaching examples supplied by the evaluator; no independent corpus or neural training'}
        print(json.dumps(report,indent=2,ensure_ascii=False))
        return report

if __name__=='__main__':run()
