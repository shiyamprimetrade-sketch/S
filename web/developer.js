'use strict';
const $ = id => document.getElementById(id);
let current = null, questions = {}, busy = false, editVersion = 0;
const message = (text, error=false) => { $('status').textContent=text; $('status').classList.toggle('error',error); };
async function api(path, body) {
  const response = await fetch('/api/developer/'+path, body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function node(tag, text) { const n=document.createElement(tag); n.textContent=text; return n; }
function lock(value) {
  busy=value;
  document.querySelectorAll('button,input,textarea').forEach(n=>n.disabled=value);
  $('export').disabled=value || !current;
}
async function action(fn) {
  if(busy)return;
  lock(true);
  try { await fn(); } catch(e) { message(e.message,true); } finally { lock(false); }
}
function answers() { return Object.fromEntries(Object.keys(questions).map(key=>[key,$('answer-'+key).value])); }
function setForm(spec) {
  current=spec;
  $('request').value=spec.intent.request; $('name').value=spec.name;
  for(const key of Object.keys(questions)) $('answer-'+key).value=spec.answers[key] || '';
  show(spec);
}
function show(spec) {
  $('kind').textContent=spec.intent.kind.toUpperCase();
  $('review-title').textContent=spec.name;
  $('revision').textContent=`Revision ${spec.revision} · ${spec.status}`;
  $('spec-json').textContent=JSON.stringify(spec,null,2);
  const summary=$('summary'); summary.replaceChildren();
  summary.append(node('h3',spec.clarifications.length ? 'Questions to resolve' : 'Requirements collected'));
  const list=node('ul','');
  for(const item of spec.clarifications) list.append(node('li',item.question));
  if(!spec.clarifications.length) list.append(node('li','No blank required answers. Architecture and acceptance quality still need review.'));
  summary.append(list,node('h3','Suggested starting stack'),node('p',
    spec.technology_proposal.languages.join(' / ') || 'Requires architecture review'),
    node('p',spec.technology_proposal.reason),node('h3','Implementation plan'));
  const plan=node('ol',''); spec.plan.forEach(x=>plan.append(node('li',x))); summary.append(plan);
  $('export').disabled=false;
}
async function projects() {
  const items=await api('projects'); $('projects').replaceChildren();
  items.forEach(p=>{ const button=node('button',p.name); button.type='button';
    button.onclick=()=>action(async()=>{
      if(editVersion && !window.confirm('Discard unsaved changes and open this project?'))return;
      const spec=await api('spec?project_id='+encodeURIComponent(p.id));
      setForm(spec); editVersion=0; await history(); message('Reopened saved project.');
    }); $('projects').append(button); });
}
async function history() {
  const pid=current.project_id;
  const items=await api('history?project_id='+encodeURIComponent(pid)); $('history').replaceChildren();
  items.forEach(spec=>{
    const button=node('button',`Restore revision ${spec.revision} · ${spec.status}`);button.type='button';
    button.onclick=()=>action(async()=>{
      if(!window.confirm(`Restore revision ${spec.revision} as a new draft checkpoint? Unsaved edits will be discarded; saved revisions remain available.`))return;
      const restored=await api('rollback',{project_id:pid,revision:spec.revision,expected_revision:current.revision,confirm:true});
      setForm(restored);editVersion=0;await history();await projects(); message('Restored as a new draft. Previous checkpoints preserved.');
    }); $('history').append(button);
  });
}
async function save(approve) {
  if(!$('requirements-form').reportValidity())return;
  if(approve && !window.confirm('Approve these requirements and save a decision in S project memory? This does not authorize execution.'))return;
  const spec=await api('spec',{name:$('name').value,request:$('request').value,answers:answers(),
    project_id:current ? current.project_id : null,expected_revision:current ? current.revision : 0,approve});
  current=spec;show(spec);editVersion=0;await history();await projects();
  message(`Saved revision ${spec.revision}. ${spec.clarifications.length} unanswered clarification(s).`);
}
$('requirements-form').oninput=()=>{editVersion++;message('Unsaved changes. Save a checkpoint to preserve them.');};
$('requirements-form').onsubmit=e=>{e.preventDefault();action(()=>save(false));};
$('approve').onclick=()=>action(()=>save(true));
$('route').onclick=()=>action(async()=>{ const intent=await api('intent',{request:$('request').value}); $('kind').textContent=intent.kind.toUpperCase(); message('Intent recognized. Complete the requirements below.'); });
$('new-project').onclick=()=>action(async()=>{
  if(editVersion && !window.confirm('Discard unsaved changes and start a new project?'))return;
  current=null;editVersion=0;$('requirements-form').reset();$('review-title').textContent='Your blueprint starts here';
  $('revision').textContent='No saved checkpoint';$('summary').textContent='Describe and save your requirements.';
  $('history').textContent='No checkpoints yet.';$('spec-json').textContent='{}';$('kind').textContent='APPLICATION';message('New project.');
});
$('export').onclick=()=>{
  if(!current)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));
  const link=node('a','');link.href=url;link.download=`s-spec-${current.project_id}-r${current.revision}.json`;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);message('Exported the last saved checkpoint. Unsaved edits are excluded.');
};
$('show-toolchains').onclick=()=>action(async()=>{
  const data=await api('toolchains');$('catalog').replaceChildren();
  for(const tool of [...data.languages,...data.frameworks]) {
    const a=node('a',tool.name || tool.id);a.href=tool.official_source;a.target='_blank';a.rel='noopener noreferrer';
    a.append(node('span',tool.adapter_status+' · stable version not verified'));$('catalog').append(a);
  }
  $('catalog-panel').hidden=false;message('Catalog only. No installation or tool execution performed.');
});
window.addEventListener('beforeunload',event=>{if(editVersion){event.preventDefault();event.returnValue='';}});
action(async()=>{
  questions=await api('questions');
  for(const [key,question] of Object.entries(questions)) {
    const label=node('label',question);label.htmlFor='answer-'+key;
    const input=document.createElement('textarea');input.id='answer-'+key;input.maxLength=5000;input.rows=key==='features'?4:2;
    $('questions').append(label,input);
  }
  const request=new URLSearchParams(location.hash.slice(1)).get('request');
  if(request){$('request').value=request;const intent=await api('intent',{request});$('kind').textContent=intent.kind.toUpperCase();}
  await projects();
});
