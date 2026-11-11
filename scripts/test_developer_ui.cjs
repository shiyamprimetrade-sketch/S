const assert = require('node:assert/strict');
const {test} = require('node:test');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const script=fs.readFileSync(path.join(__dirname,'../web/developer.js'),'utf8');
const questions={purpose:'What should it do?',users:'Who uses it?',features:'Features?',platform:'Platform?',data:'Data?',constraints:'Constraints?',acceptance:'Checks?'};
function specification(revision=1) {
  return {project_id:'P',name:'Synthetic project',revision,status:'draft',intent:{kind:'application',request:'S develop application'},
    answers:Object.fromEntries(Object.keys(questions).map(k=>[k,'Synthetic '+k])),clarifications:[],
    technology_proposal:{languages:[],reason:'Requires review'},plan:['Review','Verify']};
}
async function fixture() {
  const nodes=new Map(), pending=[],downloads=[],dialogs=[];
  function element(tag='div') {
    const e={tag,value:'',disabled:false,textContent:'',hidden:false,children:[],classList:{toggle(){}},
      append(...v){this.children.push(...v);},replaceChildren(...v){this.children=v;},
      reportValidity(){return true;},reset(){for(const k of nodes.keys())if(k.startsWith('answer-')||k==='name')nodes.get(k).value='';el('request').value='S develop application';},
      click(){downloads.push(this);}};
    Object.defineProperty(e,'id',{get(){return this._id;},set(v){this._id=v;nodes.set(v,this);}});
    return e;
  }
  function el(id) {if(!nodes.has(id)){const e=element();e.id=id;}return nodes.get(id);}
  const context={document:{getElementById:el,createElement:element,querySelectorAll:()=>[...nodes.values()]},
    location:{hash:''}, URLSearchParams,Blob,URL:{createObjectURL:()=> 'blob:synthetic',revokeObjectURL(){}},setTimeout:()=>0,
    window:{confirm:msg=>{dialogs.push(msg);return true;},addEventListener(){}},
    fetch:(url,options)=>new Promise(resolve=>pending.push({url,options,
      reply:(body,ok=true)=>resolve({ok,json:async()=>body})}))};
  const flush=async()=>{for(let i=0;i<25;i++)await Promise.resolve();};
  vm.runInNewContext(script,context);
  pending.shift().reply(questions);await flush();pending.shift().reply([]);await flush();
  el('request').value='S develop application';el('name').value='Synthetic project';
  for(const key of Object.keys(questions))el('answer-'+key).value='Synthetic '+key;
  const submit=()=>{el('requirements-form').onsubmit({preventDefault(){}});};
  const save=async(spec=specification())=>{
    submit();pending.shift().reply(spec);await flush();pending.shift().reply([spec]);await flush();
    pending.shift().reply([{id:'P',name:spec.name}]);await flush();
  };
  return {el,pending,flush,submit,save,downloads,dialogs};
}

test('save sends structured answers and no execution permissions',async()=>{
  const f=await fixture();f.submit();
  const call=f.pending.shift(),body=JSON.parse(call.options.body);
  assert.equal(call.url,'/api/developer/spec');assert.equal(body.project_id,null);
  assert.equal(body.expected_revision,0);assert.equal(body.approve,false);
  assert.deepEqual(Object.keys(body.answers).sort(),Object.keys(questions).sort());
  assert.equal(body.permissions,undefined);
  call.reply({error:'Synthetic failure'},false);await f.flush();
  assert.equal(f.el('status').textContent,'Synthetic failure');assert.equal(f.el('save').disabled,false);
});

test('duplicate save and project changes are blocked during request',async()=>{
  const f=await fixture();f.submit();f.submit();f.el('new-project').onclick();
  assert.equal(f.pending.length,1);assert.equal(f.el('request').disabled,true);
  f.pending.shift().reply({error:'Expected failure'},false);await f.flush();
  assert.equal(f.el('request').disabled,false);
});

test('approval requires explicit confirmation and persists approval flag',async()=>{
  const f=await fixture();const job=f.el('approve').onclick();
  assert.equal(f.dialogs.length,1);assert.match(f.dialogs[0],/does not authorize execution/);
  const call=f.pending.shift();assert.equal(JSON.parse(call.options.body).approve,true);
  call.reply({error:'Missing answers'},false);await job;
  assert.equal(f.el('status').textContent,'Missing answers');
});

test('reopen populates answers and preserves checkpoint revision on next save',async()=>{
  const f=await fixture();await f.save();
  const open=f.el('projects').children[0].onclick();const saved=specification(4);
  f.pending.shift().reply(saved);await f.flush();f.pending.shift().reply([saved]);await open;
  assert.equal(f.el('answer-purpose').value,'Synthetic purpose');assert.match(f.el('revision').textContent,/Revision 4/);
  f.submit();assert.equal(JSON.parse(f.pending[0].options.body).expected_revision,4);
});

test('malicious markup renders as text and never becomes HTML',async()=>{
  const f=await fixture(),spec=specification();spec.name='<img src=x onerror=alert(1)>';
  spec.clarifications=[{field:'purpose',question:'<script>steal()</script>'}];await f.save(spec);
  assert.equal(f.el('review-title').textContent,spec.name);
  assert.equal(f.el('summary').children[1].children[0].textContent,'<script>steal()</script>');
  assert.equal(f.el('review-title').innerHTML,undefined);
});

test('rollback is confirmed, revision checked, and old approval is not resent',async()=>{
  const f=await fixture();await f.save();
  const job=f.el('history').children[0].onclick(),call=f.pending.shift();
  const body=JSON.parse(call.options.body);assert.equal(body.confirm,true);assert.equal(body.expected_revision,1);
  assert.equal(body.approve,undefined);assert.match(f.dialogs[0],/new draft checkpoint/);
  call.reply(specification(2));await f.flush();f.pending.shift().reply([specification(2),specification()]);await f.flush();
  f.pending.shift().reply([{id:'P',name:'Synthetic project'}]);await job;assert.match(f.el('revision').textContent,/Revision 2/);
});

test('new project clears checkpoint and export state',async()=>{
  const f=await fixture();await f.save();await f.el('new-project').onclick();
  assert.equal(f.el('export').disabled,true);assert.equal(f.el('spec-json').textContent,'{}');
  assert.equal(f.el('answer-purpose').value,'');f.submit();
  assert.equal(JSON.parse(f.pending[0].options.body).project_id,null);
});

test('export labels the saved checkpoint and warns that unsaved edits are excluded',async()=>{
  const f=await fixture();await f.save();f.el('answer-purpose').value='Unsaved';f.el('requirements-form').oninput();
  f.el('export').onclick();assert.equal(f.downloads[0].download,'s-spec-P-r1.json');
  assert.match(f.el('status').textContent,/Unsaved edits are excluded/);
});
