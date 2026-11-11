const assert=require('node:assert/strict');
const {test}=require('node:test');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');
const script=fs.readFileSync(path.join(__dirname,'../web/coding.js'),'utf8');
function artifact(revision=1){return {project_id:'P',spec_revision:1,revision,recipe:'python-cli',provenance:{provider:'template'},files:{'app.py':'print(1)','test_app.py':'# test'}};}
function fixture(){
  const nodes=new Map(),pending=[],messages=[],confirmations=[];
  const el=id=>{if(!nodes.has(id))nodes.set(id,{value:'',checked:false,children:[],textContent:'',append(...n){this.children.push(...n)},replaceChildren(){this.children=[]}});return nodes.get(id);};
  const c={current:{project_id:'P',revision:1,status:'approved'},editVersion:0,busy:false,
    $:el,node:(tag,text)=>({tag,textContent:text}),onProjectChange:()=>{},hasSourceEdits:()=>false,
    message:msg=>messages.push(msg),window:{confirm:msg=>{confirmations.push(msg);return true;}},
    api:(route,body)=>new Promise((resolve,reject)=>pending.push({route,body,reply:resolve,fail:reject})),
    Blob,URL:{createObjectURL:()=>'',revokeObjectURL(){}},Uint8Array,atob,setTimeout:()=>0};
  c.action=async fn=>{if(c.busy)return;c.busy=true;try{await fn()}catch(e){messages.push(e.message)}finally{c.busy=false}};
  vm.createContext(c);vm.runInContext(script,c);
  el('code-recipe').value='python-cli';el('code-provider').value='template';el('code-repairs').value='1';el('code-consent').checked=true;
  const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve()};
  const load=async()=>{const job=el('code-refresh').onclick();pending.shift().reply(artifact());await flush();pending.shift().reply([artifact()]);await flush();pending.shift().reply([]);await job;};
  return {c,el,pending,messages,confirmations,flush,load};
}
test('generation requires approved saved requirements and explicit consent',async()=>{
  const f=fixture();f.c.current.status='draft';await f.el('code-generate').onclick();assert.equal(f.pending.length,0);
  assert.match(f.messages.pop(),/Approve/);f.c.current.status='approved';f.c.editVersion=1;
  await f.el('code-generate').onclick();assert.equal(f.pending.length,0);
  f.c.editVersion=0;f.el('code-consent').checked=false;await f.el('code-generate').onclick();assert.equal(f.pending.length,0);
});
test('source generation carries scoped revision and no runtime permissions',async()=>{
  const f=fixture();const job=f.el('code-generate').onclick();f.pending.shift().reply([]);await f.flush();
  const call=f.pending.shift();assert.equal(call.route,'generate');assert.equal(call.body.project_id,'P');assert.equal(call.body.spec_revision,1);
  assert.equal(call.body.provider,'template');assert.equal(call.body.command,undefined);
  call.reply(artifact());await f.flush();f.pending.shift().reply([artifact()]);await job;
  assert.equal(f.el('code-content').value,'print(1)');assert.equal(f.el('code-consent').checked,false);
});
test('project change clears source, consent and diagnostics',async()=>{
  const f=fixture();await f.load();f.el('code-content').oninput();assert.equal(f.c.hasSourceEdits(),true);
  f.c.onProjectChange(null);assert.equal(f.c.hasSourceEdits(),false);assert.equal(f.el('code-content').value,'');
  assert.equal(f.el('code-consent').checked,false);assert.match(f.el('code-output').textContent,/No build/);
});
test('cross-project source responses cannot populate the editor',async()=>{
  const f=fixture();const job=f.el('code-refresh').onclick();const a=artifact();a.project_id='OTHER';f.pending.shift().reply(a);await job;
  assert.equal(f.el('code-content').value,'');assert.match(f.messages.pop(),/different project/);
});
test('unsaved source cannot run and repair budget cannot expand',async()=>{
  const f=fixture();await f.load();f.el('code-content').oninput();await f.el('code-build').onclick();
  assert.equal(f.pending.length,0);assert.match(f.messages.pop(),/Save source/);
  f.c.onProjectChange(null);await f.load();f.el('code-consent').checked=true;f.el('code-repairs').value='3';
  await f.el('code-build').onclick();assert.equal(f.pending.length,0);assert.match(f.messages.pop(),/0–2/);
});
test('bounded build submission is confirmed and reports actual result',async()=>{
  const f=fixture();await f.load();const job=f.el('code-build').onclick();const call=f.pending.shift();
  assert.equal(call.route,'build');assert.equal(call.body.artifact_revision,1);assert.equal(call.body.max_repairs,1);
  assert.equal(call.body.shell,undefined);assert.match(f.confirmations[0],/isolated container/);
  call.reply({artifact:artifact(),attempts:[],final_status:'blocked'});await f.flush();f.pending.shift().reply([artifact()]);await job;
  assert.match(f.el('code-output').textContent,/blocked/);assert.equal(f.el('code-consent').checked,false);
});

test('runtime probe is consented and remains a runtime-only result',async()=>{
  const f=fixture();f.c.current.status='draft';f.el('neural-prompt').value='A';f.el('code-consent').checked=false;
  await f.el('neural-probe').onclick();assert.equal(f.pending.length,0);
  f.el('code-consent').checked=true;const job=f.el('neural-probe').onclick();const call=f.pending.shift();
  assert.equal(call.route,'neural-probe');assert.equal(call.body.project_id,'P');assert.equal(call.body.tokens,16);
  assert.equal(call.body.model_path,undefined);
  call.reply({inference_completed:true,coding_capability:'unverified',runtime:'python-stdlib-f32'});await job;
  assert.match(f.el('code-output').textContent,/python-stdlib-f32/);assert.match(f.messages.pop(),/coding ability is still unverified/);
});

test('capability evaluation distinguishes independent results from generic promotion',async()=>{
  const f=fixture();f.el('evaluation-execute').checked=false;
  const job=f.el('evaluate-coding').onclick();const call=f.pending.shift();
  assert.equal(call.route,'evaluate-coding');assert.equal(call.body.execute,false);assert.equal(call.body.max_repairs,0);
  call.reply({task_count:3,final_tasks_correct:0,promotion_eligible:false});await job;
  assert.match(f.messages.pop(),/General capability is not established/);assert.equal(f.el('code-consent').checked,false);
});
