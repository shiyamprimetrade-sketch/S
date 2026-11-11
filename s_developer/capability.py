"""Independent, reproducible small coding evaluation; no model promotion shortcut.

These are public synthetic diagnostic tasks, not a secret or uncontaminated
industry benchmark. Hidden means withheld from generation/repair and containers.
"""
import hashlib
import json
import time
from .bundles import fingerprint,redact,validate_files
from .providers import provider,ContextLimitError
from .sandbox import DockerRunner

VERSION='s-public-coding-diagnostic-v1'
PROTOCOL='Return JSON {"files":{"app.py":"...","test_app.py":"..."}}. app.py must define solve(p). Use stdlib only.\n'
TASKS=(
 {'id':'positive-total','contract':'Return {"total": sum of positive integers in p["values"]}. Return zero for an empty list.',
  'cases':[
   {'split':'public','input':{'values':[3,-1,2,0]},'expected':{'total':5}},
   {'split':'withheld','input':{'values':[-9,1,1,8]},'expected':{'total':10}},
   {'split':'withheld','input':{'values':[]},'expected':{'total':0}}]},
 {'id':'unique-words','contract':'Return {"words": sorted unique lowercase words from p["text"].split()}.',
  'cases':[
   {'split':'public','input':{'text':'Beta alpha BETA'},'expected':{'words':['alpha','beta']}},
   {'split':'withheld','input':{'text':' Zebra  APPLE\tapple\n'},'expected':{'words':['apple','zebra']}},
   {'split':'withheld','input':{'text':''},'expected':{'words':[]}}]},
 {'id':'palindrome','contract':'Return {"palindrome": a JSON boolean} for p["text"], ignoring ASCII spaces and letter case.',
  'cases':[
   {'split':'public','input':{'text':'A b a'},'expected':{'palindrome':True}},
   {'split':'withheld','input':{'text':'hello'},'expected':{'palindrome':False}},
   {'split':'withheld','input':{'text':'Never odd or even'},'expected':{'palindrome':True}}]},
)


def canonical(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def _no_constant(value):raise ValueError('Nonfinite JSON output')


def judge(result,expected):
    if result.get('status')=='blocked':return {'status':'blocked','passed':False,'diagnostic':result.get('diagnostic','Runtime blocked')}
    if result.get('status')!='toolchain_passed':return {'status':'runtime_failed','passed':False,'diagnostic':redact(result.get('output','Execution failed'))}
    try:
        actual=json.loads(result.get('output',''),parse_constant=_no_constant)
        correct=canonical(actual)==canonical(expected)
        return {'status':'passed' if correct else 'wrong_answer','passed':correct,
                'actual':actual,'diagnostic':'JSON result matched the host oracle' if correct else 'JSON result did not satisfy this case'}
    except (ValueError,TypeError):return {'status':'invalid_output','passed':False,'diagnostic':'Expected one finite JSON result without extra stdout'}


class CapabilityEvaluator:
    def __init__(self,config):self.config=config
    def _generate(self,engine,task):
        prompt=PROTOCOL+task['contract']
        if getattr(engine,'id',None) in ('native','ollama-external'):
            return engine._generate(prompt,'python-cli')
        # The deterministic template baseline is intentionally not specialized
        # to these tasks. Correct-code fixtures are test-only injected providers.
        spec={'name':task['id'],'answers':{'purpose':task['contract']},'functional_requirements':[task['contract']]}
        return engine.generate(spec,'python-cli')
    def evaluate(self,identifier,consent=False,execute=False,max_repairs=0):
        if consent is not True:raise PermissionError('Explicit evaluation consent required')
        if type(execute) is not bool or type(max_repairs) is not int or not 0<=max_repairs<=1:
            raise ValueError('Evaluation supports a boolean execution flag and 0–1 repair')
        if execute and self.config.get('execution_enabled') is not True:raise PermissionError('Host must enable isolated execution')
        engine=provider(identifier,self.config);provenance=engine.provenance()
        start=time.monotonic();tasks=[]
        for task in TASKS:
            row={'id':task['id'],'prompt_sha256':hashlib.sha256((PROTOCOL+task['contract']).encode()).hexdigest(),
                 'format_passed':False,'initial_correct':False,'final_correct':False,'attempts':[]}
            tasks.append(row)
            try:files=validate_files(self._generate(engine,task),'python-cli');row['format_passed']=True
            except Exception as error:
                row.update(status='input_context_blocked' if isinstance(error,ContextLimitError) else 'generation_failed',
                           diagnostic=redact(type(error).__name__+': '+str(error)));continue
            row['source_sha256']=fingerprint(files)
            if not execute:row['status']='execution_not_assessed';continue
            for attempt in range(max_repairs+1):
                row.setdefault('source_history',[]).append({'source_sha256':fingerprint(files),'files':dict(files)})
                results=[]
                for index,case in enumerate(task['cases']):
                    try:raw=DockerRunner(self.config.get('images',{})).run(files,'python-cli',timeout=10,evaluation_input=case['input'])
                    except Exception as error:raw={'status':'blocked','diagnostic':redact(type(error).__name__+': '+str(error))}
                    verdict=judge(raw,case['expected'])
                    # Expected answers are never passed to the runtime or provider.
                    results.append({'case_index':index,'split':case['split'],'verdict':verdict,
                                    'returncode':raw.get('returncode'),'image':raw.get('image')})
                    if verdict['status']=='blocked':break
                correct=len(results)==len(task['cases']) and all(r['verdict']['passed'] for r in results)
                row['attempts'].append({'attempt':attempt,'source_sha256':fingerprint(files),'cases':results,'passed':correct})
                if attempt==0:row['initial_correct']=correct
                row['final_correct']=correct
                if correct:row['status']='passed';break
                if any(r['verdict']['status']=='blocked' for r in results):row['status']='runtime_blocked';break
                row['status']='failed_independent_cases'
                if attempt==max_repairs:break
                public=results[0]
                # Repair receives only public diagnostic/input/answer; withheld
                # case inputs, expected results and observations never enter it.
                diagnostic=canonical({'public_input':task['cases'][0]['input'],
                    'public_expected':task['cases'][0]['expected'],'public_result':public['verdict']})
                if public['verdict']['passed']:
                    diagnostic='Public case passed. Review the task contract and general edge cases; withheld details are not disclosed.'
                spec={'name':task['id'],'answers':{'purpose':task['contract']},'functional_requirements':[task['contract']]}
                try:
                    repaired=validate_files(engine.repair(spec,'python-cli',files,diagnostic),'python-cli')
                    if fingerprint(repaired)==fingerprint(files):raise ValueError('Repair made no source change')
                    files=repaired
                except Exception as error:row.update(status='repair_stopped',diagnostic=redact(type(error).__name__+': '+str(error)));break
        format_count=sum(r['format_passed'] for r in tasks)
        initial_count=sum(r['initial_correct'] for r in tasks);final_count=sum(r['final_correct'] for r in tasks)
        assessed=sum(any(len(a['cases'])==len(TASKS[i]['cases']) and all(c['verdict']['status']!='blocked' for c in a['cases'])
                         for a in row['attempts']) for i,row in enumerate(tasks))
        return {'schema_version':1,'benchmark':VERSION,'provider':provenance,'tasks':tasks,
                'format_passed':format_count,'initial_tasks_correct':initial_count,'final_tasks_correct':final_count,
                'task_count':len(tasks),'execution_requested':execute,'elapsed_seconds':round(time.monotonic()-start,3),
                'repair_budget':max_repairs,'case_timeout_seconds':10,
                'tasks_fully_assessed':assessed,
                'suite_passed':execute and final_count==len(tasks),
                'promotion_eligible':False,
                'scope':'Three public synthetic Python-function diagnostics only; not general coding/business capability',
                'integrity':'Expected answers compared outside generated containers; withheld cases excluded from generation and repair'}
