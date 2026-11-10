"""A tiny, disclosed, hand-authored example — not a real-world accuracy claim."""
import json
import tempfile
from pathlib import Path
from continuum.apprenticeship import Apprenticeship
from continuum.meaning import TeacherMemory
from continuum.storage import Store


def main():
    with tempfile.TemporaryDirectory() as temp:
        store=Store(Path(temp)/'evaluation.sqlite3')
        pid=store.create_project('Disposable synthetic evaluation')['id']
        teacher=TeacherMemory(store)
        benchmark=Apprenticeship(store)
        lesson=teacher.teach(pid,'Investigate first',
                             'Check assumptions and measure evidence before deciding.',
                             ['Apply rigorous analysis to inventions'])
        benchmark.add_case(pid,'Undertake careful investigation of a prototype',lesson['id'])
        benchmark.add_case(pid,'Bake a cinnamon cake for breakfast',None)
        before=benchmark.run(pid)['metrics']
        teacher.define_concept(pid,'reasoning effort',
                               ['rigorous analysis','careful investigation'])
        after=benchmark.run(pid)['metrics']
        result={'kind':'designer-authored demonstration, NOT general ability measurement',
                'cases':2,'before':before,'after':after}
        print(json.dumps(result,indent=2,ensure_ascii=False))
        assert before['taught_meaning_correct']==1
        assert after['baseline_correct']==1
        assert after['taught_meaning_correct']==2


if __name__=='__main__':main()
