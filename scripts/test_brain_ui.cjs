const assert = require('node:assert/strict');
const {test} = require('node:test');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const script = fs.readFileSync(path.join(__dirname, '../web/brain.js'), 'utf8');

function element() {
  return {value:'', checked:false, disabled:false, readOnly:false, textContent:'', children:[],
    append(...nodes){this.children.push(...nodes);}, replaceChildren(...nodes){this.children=nodes;}};
}
async function fixture() {
  const nodes = new Map();
  const el = id => {if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id);};
  const pending = [];
  const context = {document:{getElementById:el, createElement:element}, URLSearchParams,
    confirm:()=>true, fetch:(url, options)=>new Promise(resolve=>pending.push({url, options,
      reply:body=>resolve({ok:true,json:async()=>body})}))};
  vm.runInNewContext(script, context);
  const flush = async()=>{for(let i=0;i<12;i++) await Promise.resolve();};
  pending.shift().reply([{id:'A', name:'Synthetic A'}]); await flush();
  pending.shift().reply({available:true,memory_enabled:true,tools_enabled:true,metrics:{}}); await flush();
  el('scope').value='A'; el('intents').value='[]'; el('question').value='Synthetic question';
  const change = id=>{el('scope').value=id; el('scope').onchange();};
  return {el,pending,flush,change};
}

test('late memory response cannot populate a different project', async()=>{
  const f=await fixture(); const job=f.el('list').onclick();
  f.change('B');
  f.pending.shift().reply([{id:'old',content:'A-only synthetic content',source:'fixture'}]);
  await job;
  assert.equal(f.el('records').children.length,0);
});

test('latest search wins when responses arrive backwards', async()=>{
  const f=await fixture(); f.el('query').value='old'; const first=f.el('search').onclick();
  f.el('query').value='new'; const second=f.el('search').onclick();
  const old=f.pending.shift(), current=f.pending.shift();
  current.reply([{id:'new',content:'new result'}]); await second;
  old.reply([{id:'old',content:'old result'}]); await first;
  assert.match(f.el('records').children[0].children[0].textContent,/new result/);
});

test('double send cannot create duplicate sessions', async()=>{
  const f=await fixture(); const first=f.el('send').onclick(); const second=f.el('send').onclick();
  assert.equal(f.pending.length,1);
  assert.equal(f.el('send').disabled,true);
  f.pending.shift().reply({id:'session-A'}); await f.flush();
  f.pending.shift().reply({content:'synthetic reply',brain:{}}); await Promise.all([first,second]);
});

test('changing project during session creation cancels submission and stale session reuse', async()=>{
  const f=await fixture(); const first=f.el('send').onclick(); f.change('B');
  f.pending.shift().reply({id:'session-A'}); await f.flush();
  assert.equal(f.pending.length,0);
  await first;
  const second=f.el('send').onclick();
  assert.equal(f.pending[0].url,'/api/sessions');
  assert.equal(JSON.parse(f.pending[0].options.body).project_id,'B');
  f.pending.shift().reply({id:'session-B'}); await f.flush();
  f.pending.shift().reply({content:'B reply',brain:{}}); await second;
});

test('late answer cannot appear after switching projects, even when switching back', async()=>{
  const f=await fixture(); const job=f.el('send').onclick();
  f.pending.shift().reply({id:'session-A'}); await f.flush();
  f.change('B'); f.change('A');
  f.pending.shift().reply({content:'stale reply',brain:{}}); await job;
  assert.equal(f.el('answer').textContent,'');
});

test('memory editing preserves source and expiry instead of offering ignored changes', async()=>{
  const f=await fixture(); const job=f.el('list').onclick();
  f.pending.shift().reply([{id:'memory',content:'synthetic',source:'fixture',expires_at:Math.floor(Date.now()/1000)+600}]);
  await job; f.el('records').children[0].children[1].onclick();
  assert.equal(f.el('source').readOnly,true); assert.equal(f.el('ttl').disabled,true);
  f.el('clear').onclick();
  assert.equal(f.el('source').readOnly,false); assert.equal(f.el('ttl').disabled,false);
  assert.equal(f.el('ttl').value,'86400');
});

test('status refresh cannot re-enable Send while a request is pending', async()=>{
  const f=await fixture(); const chat=f.el('send').onclick();
  const refresh=f.el('refresh').onclick();
  f.pending.pop().reply({available:true,memory_enabled:true,tools_enabled:true,metrics:{}});
  await refresh; assert.equal(f.el('send').disabled,true);
  f.pending.shift().reply({id:'session-A'}); await f.flush();
  f.pending.shift().reply({content:'reply',brain:{}}); await chat;
  assert.equal(f.el('send').disabled,false);
});

test('duplicate memory save is suppressed and old completion preserves new project editor', async()=>{
  const f=await fixture(); f.el('consent').checked=true; f.el('content').value='A synthetic';
  const first=f.el('save').onclick(), duplicate=f.el('save').onclick();
  assert.equal(f.pending.length,1);
  f.change('B'); f.el('content').value='B draft';
  f.pending.shift().reply({id:'saved-A'}); await Promise.all([first,duplicate]);
  assert.equal(f.el('content').value,'B draft'); assert.equal(f.pending.length,0);
});

test('memory edit submits only supported fields', async()=>{
  const f=await fixture(); f.el('memory-id').value='existing'; f.el('consent').checked=true;
  f.el('content').value='revised'; const job=f.el('save').onclick();
  const body=JSON.parse(f.pending[0].options.body);
  assert.equal(body.source,undefined); assert.equal(body.ttl_seconds,undefined);
  f.pending.shift().reply({id:'existing'}); await f.flush();
  f.pending.shift().reply([]); await job;
});
