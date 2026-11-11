"""Project-scoped source checkpoints and permissioned bounded development loop."""
import base64
import io
import json
import threading
import zipfile
from continuum.storage import now
from .bundles import RECIPES, fingerprint, redact, validate_files
from .providers import provider
from .sandbox import DockerRunner

SCHEMA='''
CREATE TABLE IF NOT EXISTS developer_artifacts(
 project_id TEXT NOT NULL REFERENCES projects(id), revision INTEGER NOT NULL,
 document TEXT NOT NULL, PRIMARY KEY(project_id,revision));
CREATE TABLE IF NOT EXISTS developer_runs(
 id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL REFERENCES projects(id), document TEXT NOT NULL);
'''


class DevelopmentEngine:
    def __init__(self,specifications,config=None):
        self.specs=specifications;self.store=specifications.store;self.config=config or {};self.lock=threading.RLock()
        with self.store.connect() as db:db.executescript(SCHEMA)
    def capabilities(self):
        return {'recipes':RECIPES,'providers':{'template':True,'native':bool(self.config.get('native_checkpoint')),
                 'ollama-external':bool(self.config.get('ollama_model'))},
                'execution_enabled':self.config.get('execution_enabled') is True,
                'runtime':'digest-pinned Docker; no host execution fallback',
                'images_prepared':sorted(self.config.get('images',{})),
                'business_acceptance':'requires independent review beyond toolchain tests',
                'model_installation':'operator CLI only; never triggered by project instructions'}
    def _spec(self,pid,expected):
        spec=self.specs.latest(pid)
        if type(expected) is not int or spec['revision']!=expected:raise ValueError('Specification changed; reopen it')
        if spec['status']!='approved':raise PermissionError('Approve the complete specification before generation or execution')
        return spec
    def _body(self,body,allowed):
        if not isinstance(body,dict) or set(body)!=set(allowed):raise ValueError('Unexpected development request fields')
        if body.get('consent') is not True:raise PermissionError('Explicit action approval required')
    def artifacts(self,pid):
        self.store.get_project(pid)
        with self.store.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT document FROM developer_artifacts WHERE project_id=? ORDER BY revision DESC',(pid,))]
    def latest(self,pid):
        rows=self.artifacts(pid)
        if not rows:raise ValueError('Generate a source checkpoint first')
        return rows[0]
    def runs(self,pid):
        self.store.get_project(pid)
        with self.store.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT document FROM developer_runs WHERE project_id=? ORDER BY id DESC LIMIT 100',(pid,))]
    def _record(self,pid,value):
        value=dict(value,created_at=now())
        with self.store.connect() as db:
            db.execute('INSERT INTO developer_runs(project_id,document) VALUES (?,?)',(pid,json.dumps(value)))
        return value
    def _save(self,spec,recipe,files,provenance,expected):
        files=validate_files(files,recipe)
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT document FROM developer_specs WHERE project_id=? ORDER BY revision DESC LIMIT 1',(spec['project_id'],)).fetchone()
            latest=json.loads(row[0]) if row else {}
            if latest.get('revision')!=spec['revision'] or latest.get('status')!='approved':
                raise ValueError('Specification changed during development; no source replaced')
            revision=db.execute('SELECT COALESCE(MAX(revision),0) FROM developer_artifacts WHERE project_id=?',(spec['project_id'],)).fetchone()[0]
            if type(expected) is not int or expected!=revision:raise ValueError('Source checkpoint changed; reopen before writing')
            document={'project_id':spec['project_id'],'revision':revision+1,'spec_revision':spec['revision'],
                      'recipe':recipe,'files':files,'source_sha256':fingerprint(files),'provenance':provenance,
                      'created_at':now(),'status':'generated-unverified'}
            db.execute('INSERT INTO developer_artifacts VALUES (?,?,?)',(spec['project_id'],revision+1,json.dumps(document)))
        return document
    def generate(self,body):
        self._body(body,('project_id','spec_revision','expected_artifact_revision','recipe','provider','consent'))
        with self.lock:
            spec=self._spec(body['project_id'],body['spec_revision'])
            if body['recipe'] not in RECIPES:raise ValueError('Unknown recipe')
            engine=provider(body['provider'],self.config)
            provenance={'provider':body['provider']}
            try:
                provenance=engine.provenance()
                files=engine.generate(spec,body['recipe'])
            except Exception as error:
                self._record(spec['project_id'],{'stage':'generation','status':'failed','provider':provenance,
                             'diagnostic':redact(type(error).__name__+': '+str(error))})
                raise ValueError('Generation failed; no source replaced: '+redact(str(error))) from error
            result=self._save(spec,body['recipe'],files,provenance,body['expected_artifact_revision'])
            self._record(spec['project_id'],{'stage':'generation','status':'source_created','artifact_revision':result['revision'],
                         'source_sha256':result['source_sha256'],'provider':provenance})
            return result
    def edit(self,body):
        self._body(body,('project_id','expected_artifact_revision','path','content','consent'))
        with self.lock:
            current=self.latest(body['project_id']);spec=self._spec(body['project_id'],current['spec_revision'])
            if body['path'] not in current['files']:raise ValueError('Select an existing source file')
            files=dict(current['files']);files[body['path']]=body['content']
            return self._save(spec,current['recipe'],files,{'provider':'user-edit','based_on':current['revision']},body['expected_artifact_revision'])
    def restore(self,body):
        self._body(body,('project_id','revision','expected_artifact_revision','consent'))
        with self.lock:
            old=next((a for a in self.artifacts(body['project_id']) if a['revision']==body['revision']),None)
            if old is None:raise ValueError('Unknown source checkpoint')
            spec=self._spec(body['project_id'],old['spec_revision'])
            return self._save(spec,old['recipe'],old['files'],{'provider':'rollback','based_on':old['revision']},body['expected_artifact_revision'])
    def build(self,body):
        self._body(body,('project_id','artifact_revision','repair_provider','max_repairs','consent'))
        if self.config.get('execution_enabled') is not True:raise PermissionError('Builds are disabled by the local host operator')
        maximum=body['max_repairs']
        if type(maximum) is not int or not 0<=maximum<=2:raise ValueError('Repair budget must be 0–2')
        with self.lock:
            current=self.latest(body['project_id']);spec=self._spec(body['project_id'],current['spec_revision'])
            if type(body['artifact_revision']) is not int or current['revision']!=body['artifact_revision']:
                raise ValueError('Source changed; approve the current checkpoint')
            repair=provider(body['repair_provider'],self.config) if maximum else None
            runner=DockerRunner(self.config.get('images',{}));attempts=[]
            for index in range(maximum+1):
                # Snapshot reference is immutable; source cannot be swapped by an instruction.
                try:result=runner.run(current['files'],current['recipe'])
                except Exception as error:
                    result={'status':'blocked','diagnostic':redact(type(error).__name__+': '+str(error))}
                event=self._record(spec['project_id'],dict(stage='build',attempt=index,
                    artifact_revision=current['revision'],source_sha256=current['source_sha256'],result=result))
                attempts.append(event)
                if result['status'] in ('toolchain_passed','blocked') or index==maximum:break
                try:
                    files=repair.repair(spec,current['recipe'],current['files'],result.get('output',''))
                    if fingerprint(files)==current['source_sha256']:raise ValueError('Repair produced no source change')
                    current=self._save(spec,current['recipe'],files,dict(repair.provenance(),repair_of=current['revision']),current['revision'])
                except Exception as error:
                    attempts.append(self._record(spec['project_id'],{'stage':'repair','status':'stopped',
                        'diagnostic':redact(type(error).__name__+': '+str(error))}));break
            return {'attempts':attempts,'artifact':current,'final_status':attempts[-1].get('result',{}).get('status','repair_stopped'),
                    'acceptance':'Toolchain checks only; approved business requirements still require independent verification'}
    def archive(self,pid):
        artifact=self.latest(pid);files=validate_files(artifact['files'],artifact['recipe'])
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as out:
            for name,content in files.items():out.writestr(name,content)
            out.writestr('S-PROVENANCE.json',json.dumps({k:v for k,v in artifact.items() if k!='files'},indent=2))
        return {'filename':'s-source-'+artifact['source_sha256'][:12]+'.zip',
                'archive_base64':base64.b64encode(stream.getvalue()).decode()}
    def request(self,verb,path,query,body):
        if verb=='GET':
            pid=query.get('project_id',[''])[0]
            if path=='/api/developer/capabilities':return self.capabilities()
            if path=='/api/developer/artifact':return self.latest(pid)
            if path=='/api/developer/artifacts':return [{k:v for k,v in a.items() if k!='files'} for a in self.artifacts(pid)]
            if path=='/api/developer/runs':return self.runs(pid)
            if path=='/api/developer/archive':return self.archive(pid)
        if verb=='POST':
            routes={'/api/developer/generate':self.generate,'/api/developer/edit':self.edit,
                    '/api/developer/restore-source':self.restore,'/api/developer/build':self.build}
            if path in routes:return routes[path](body)
        raise ValueError('Unknown developer engine route')
