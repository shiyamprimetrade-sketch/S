'use strict';
/* S local-first interface. No CDN, cloud API, fake search or mock assistant. */
const $=id=>document.getElementById(id);
const state={projects:[],projectId:null,sessions:[],sessionId:null,models:[],model:'s-research:local',feedback:{},busy:false,controller:null,panel:'chat',attachment:'',toastTimer:null,inventionId:null};
const modelCatalog={
 's-research:local':{name:'S Research',description:'Independent rule-based research · always offline',symbol:'S'},
 'teacherlab:local':{name:'S Mini',description:'Original small teaching-pattern classifier · offline',symbol:'◈'},
 'charllm:local':{name:'S Original',description:'From-scratch tiny byte-token text generator · experimental',symbol:'Σ'}
};
function notify(message){const t=$('toast');t.textContent=String(message);t.classList.remove('hidden');clearTimeout(state.toastTimer);state.toastTimer=setTimeout(()=>t.classList.add('hidden'),5400)}
async function api(path,body,signal){const opt={cache:'no-store',credentials:'same-origin',signal};if(body!==undefined){opt.method='POST';opt.headers={'Content-Type':'application/json'};opt.body=JSON.stringify(body)}const response=await fetch(path,opt);let data;try{data=await response.json()}catch{throw Error('Local API did not return JSON')}if(!response.ok)throw Error(data.error||`Request failed: ${response.status}`);return data}
function q(v){return new URLSearchParams(v).toString()}
function modelInfo(name){return modelCatalog[name]||{name:`Local model · ${name}`,description:'Separately installed Ollama model (not S-trained)',symbol:'O'}}
function download(content,name,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3500)}
function dateToday(iso){return (new Date(iso)).toDateString()===(new Date()).toDateString()}
function setTheme(dark){document.documentElement.classList.toggle('dark',dark);localStorage.setItem('s-theme',dark?'dark':'light')}
function switchPanel(name){if(name!=='settings'&&state.panel==='settings')clearDisplayedApiKey();state.panel=name;for(const id of ['chat','archive','knowledge','invention','settings'])$(id+'-panel').classList.toggle('hidden',id!==name);if(name==='archive')$('search-query').focus()}
function sidebar(open){const aside=$('sidebar');aside.classList.toggle('open',open);$('mobile-overlay').classList.toggle('hidden',!open)}
function popup(id,visible){$(id).classList.toggle('hidden',!visible)}
function renderModels(){const menu=$('model-menu');menu.replaceChildren();for(const model of state.models){const item=document.createElement('button');item.className='model-option'+(state.model===model?' selected':'');item.type='button';item.setAttribute('role','menuitemradio');item.setAttribute('aria-checked',String(state.model===model));const marker=document.createElement('span');marker.className='model-symbol';marker.textContent=modelInfo(model).symbol;const labels=document.createElement('span');labels.className='model-info';const title=document.createElement('strong');title.textContent=modelInfo(model).name;const sub=document.createElement('small');sub.textContent=modelInfo(model).description;labels.append(title,sub);item.append(marker,labels);item.addEventListener('click',()=>{state.model=model;refreshModelTitle();popup('model-menu',false);$('model-button').setAttribute('aria-expanded','false');renderModels()});menu.append(item)}if(!state.models.includes('charllm:local')){const item=document.createElement('button');item.className='model-option';item.disabled=true;item.textContent='S Original · Train checkpoint and install PyTorch to enable';menu.append(item)}}
function refreshModelTitle(){const name=modelInfo(state.model).name;const title=$('model-title');title.replaceChildren();const a=document.createElement('span');a.textContent='S ';const b=document.createElement('span');b.className='model-suffix';b.textContent=name.replace(/^S\s*/,'');title.append(a,b)}
async function health(){const status=$('status-label');try{const h=await api('/api/health');state.models=h.models;state.model=state.models.includes(state.model)?state.model:h.default_model;refreshModelTitle();renderModels();status.replaceChildren();const dot=document.createElement('span');dot.className='status-dot ok';status.append(dot,document.createTextNode(' Local · Ready'));$('model-details').textContent=`Available: ${state.models.map(v=>modelInfo(v).name).join(', ')}. Data: ${h.data_file}. No cloud API calls.`}catch(e){status.textContent='Local server unavailable';$('model-details').textContent=e.message;notify(e.message)}}
async function refreshProjects(preferred){state.projects=await api('/api/projects');if(!state.projects.length){const p=await api('/api/projects',{name:'S — Human Computer'});state.projects=[p]}state.projectId=(preferred&&state.projects.some(p=>p.id===preferred))?preferred:(state.projects.some(p=>p.id===state.projectId)?state.projectId:state.projects[0].id);const sel=$('project');sel.replaceChildren();for(const p of state.projects){const opt=document.createElement('option');opt.value=p.id;opt.textContent=p.name;sel.append(opt)}sel.value=state.projectId;await refreshSessions()}
async function refreshSessions(){state.sessions=await api('/api/sessions?'+q({project_id:state.projectId}));const ratings=await api('/api/feedback?'+q({project_id:state.projectId}));state.feedback=Object.fromEntries(ratings.map(x=>[x.message_id,x.rating]));renderHistory();if(state.sessionId&&!state.sessions.some(s=>s.id===state.sessionId))state.sessionId=null;if(state.sessionId){await refreshMessages()}else{showWelcome()}}
function renderHistory(){for(const id of ['history-today','history-past'])$(id).replaceChildren();for(const c of state.sessions){const parent=$(dateToday(c.created_at)?'history-today':'history-past');const row=document.createElement('div');row.className='history-item'+(c.id===state.sessionId?' active':'');const open=document.createElement('button');open.className='history-open';open.textContent=c.title;open.title=c.title;open.addEventListener('click',()=>loadChat(c.id));const del=document.createElement('button');del.className='history-delete';del.textContent='×';del.setAttribute('aria-label',`Delete ${c.title}`);del.title='Delete this conversation';del.addEventListener('click',async e=>{e.stopPropagation();if(!confirm(`Delete this local conversation?\n\n${c.title}\n\nThis cannot be undone without a backup.`))return;try{await api('/api/delete-session',{session_id:c.id});if(state.sessionId===c.id)state.sessionId=null;await refreshSessions();notify('Conversation deleted')}catch(err){notify(err.message)}});row.append(open,del);parent.append(row)}}
function showWelcome(){$('welcome-screen').classList.remove('hidden');$('messages-list').classList.add('hidden');$('messages-list').replaceChildren();state.sessionId=null;renderHistory()}
async function loadChat(id){state.sessionId=id;switchPanel('chat');sidebar(false);await refreshMessages();renderHistory()}
async function refreshMessages(){if(!state.sessionId){showWelcome();return}const msgs=await api('/api/messages?'+q({session_id:state.sessionId}));$('welcome-screen').classList.add('hidden');const list=$('messages-list');list.classList.remove('hidden');list.replaceChildren();for(const m of msgs)appendMessage(m);scrollEnd()}
function scrollEnd(){const c=$('messages-scroller');c.scrollTop=c.scrollHeight}
function codeBlock(code,lang){const box=document.createElement('div');box.className='code-block';const bar=document.createElement('div');bar.className='code-bar';const label=document.createElement('span');label.textContent=lang||'text';const copy=document.createElement('button');copy.textContent='Copy code';copy.addEventListener('click',()=>navigator.clipboard.writeText(code).then(()=>notify('Code copied')).catch(()=>notify('Clipboard unavailable')));bar.append(label,copy);const pre=document.createElement('pre');const el=document.createElement('code');el.textContent=code;pre.append(el);box.append(bar,pre);return box}
function safeContent(container,content){const regex=/```([a-zA-Z0-9_-]*)\s*\n([\s\S]*?)```/g;let pos=0,m;while((m=regex.exec(content))!==null){if(m.index>pos){const text=document.createElement('div');text.className='prose-text';text.textContent=content.slice(pos,m.index);container.append(text)}container.append(codeBlock(m[2],m[1]));pos=regex.lastIndex}if(pos<content.length){const text=document.createElement('div');text.className='prose-text';text.textContent=content.slice(pos)}if(pos<content.length)container.append(text);if(!content)container.textContent='(empty)'}
function appendMessage(m){const list=$('messages-list');const row=document.createElement('div');row.className='message-row '+m.role;if(m.role==='user'){const bubble=document.createElement('div');bubble.className='bubble-user';bubble.textContent=m.content;row.append(bubble)}else{const mark=document.createElement('span');mark.className='assistant-avatar';mark.textContent='S';const answer=document.createElement('div');answer.className='answer';const content=document.createElement('div');safeContent(content,m.content);answer.append(content);if(m.model){const meta=document.createElement('div');meta.className='model-meta';meta.textContent=modelInfo(m.model).name+' · '+(m.elapsed_ms??'?')+' ms';answer.append(meta)}const actions=document.createElement('div');actions.className='answer-actions';const copy=document.createElement('button');copy.textContent='Copy';copy.addEventListener('click',()=>navigator.clipboard.writeText(m.content).then(()=>notify('Response copied')).catch(()=>notify('Clipboard unavailable')));actions.append(copy);for(const [rating,label] of [[1,'Good'],[-1,'Poor']]){const btn=document.createElement('button');btn.textContent=rating===1?'👍':'👎';btn.title=label+' response · save feedback';if(state.feedback[m.id]===rating)btn.classList.add('selected');btn.addEventListener('click',async()=>{try{await api('/api/feedback',{message_id:m.id,rating});state.feedback[m.id]=rating;await refreshMessages();notify('Feedback saved locally')}catch(e){notify(e.message)}});actions.append(btn)}const retry=document.createElement('button');retry.textContent='Retry';retry.title='Retry by adding a new input to conversation';retry.addEventListener('click',async()=>{const msgs=await api('/api/messages?'+q({session_id:state.sessionId}));const prior=msgs.slice(0,msgs.findIndex(x=>x.id===m.id)).filter(x=>x.role==='user').at(-1);if(prior){$('chat-input').value=prior.content;resizeInput();$('chat-input').focus();notify('Original prompt copied to composer. Send to retry.')}});actions.append(retry);answer.append(actions);row.append(mark,answer)}list.append(row)}
function resizeInput(){const t=$('chat-input');t.style.height='auto';t.style.height=Math.min(t.scrollHeight,170)+'px';$('send-btn').disabled=!t.value.trim()||state.busy}
function setBusy(yes){state.busy=yes;$('send-btn').classList.toggle('hidden',yes);$('stop-btn').classList.toggle('hidden',!yes);resizeInput()}
async function send(){if(state.busy)return;const input=$('chat-input');const text=input.value.trim();if(!text)return;input.value='';resizeInput();setBusy(true);const controller=new AbortController();state.controller=controller;try{if(!state.sessionId){const s=await api('/api/sessions',{project_id:state.projectId,title:text.slice(0,70)},controller.signal);state.sessionId=s.id}switchPanel('chat');$('welcome-screen').classList.add('hidden');$('messages-list').classList.remove('hidden');const temp=document.createElement('div');temp.className='message-row user';const bub=document.createElement('div');bub.className='bubble-user';bub.textContent=text;temp.append(bub);$('messages-list').append(temp);scrollEnd();await api('/api/chat',{session_id:state.sessionId,text,model:state.model},controller.signal);await refreshSessions();await refreshMessages()}catch(e){notify(controller.signal.aborted?'Stopped waiting. The local operation may still finish.':e.message);if(state.sessionId)await refreshMessages().catch(()=>{})}finally{state.controller=null;setBusy(false);input.focus()}}
async function openArchive(){switchPanel('archive');sidebar(false)}
async function performSearch(){const term=$('search-query').value.trim();const container=$('search-results');container.replaceChildren();if(term.length<2){notify('Enter a meaningful search term');return}try{const hits=await api('/api/search?'+q({project_id:state.projectId,q:term}));if(!hits.length){container.textContent='No matching messages in this project';return}for(const h of hits){const article=document.createElement('article');article.className='result';const title=document.createElement('button');title.textContent=h.title;title.addEventListener('click',()=>loadChat(h.session_id));const label=document.createElement('div');label.className='source';label.textContent=`${h.role} · ${h.created_at?.slice(0,10)}`;const p=document.createElement('p');p.textContent=h.content;article.append(title,label,p);container.append(article)}}catch(e){notify(e.message)}}
async function refreshNotes(){const wrap=$('notes');wrap.replaceChildren();try{const notes=await api('/api/notes?'+q({project_id:state.projectId}));for(const n of notes){const item=document.createElement('article');item.className='note';const title=document.createElement('strong');title.textContent=n.title;const kind=document.createElement('small');kind.textContent=n.kind.toUpperCase()+(n.evidence?' · '+n.evidence:'');const body=document.createElement('p');body.textContent=n.body;item.append(title,document.createElement('br'),kind,body);wrap.append(item)}if(!notes.length)wrap.textContent='No saved lessons yet.'}catch(e){notify(e.message)}}
async function showKnowledge(){switchPanel('knowledge');await refreshNotes()}
function openDialog(title,fields,onSave){$('dialog-title').textContent=title;const area=$('dialog-fields');area.replaceChildren();for(const f of fields){const label=document.createElement('label');label.textContent=f.label;label.htmlFor='dialog-'+f.id;let input;if(f.options){input=document.createElement('select');for(const op of f.options){const opt=document.createElement('option');opt.value=op;opt.textContent=op;input.append(opt)}}else{input=document.createElement(f.long?'textarea':'input');if(!f.long)input.type='text'}input.id='dialog-'+f.id;input.value=f.initial||'';label.append(input);area.append(label)}$('dialog-backdrop').classList.remove('hidden');$('dialog-save').onclick=async()=>{const values={};for(const f of fields)values[f.id]=$('dialog-'+f.id).value;try{await onSave(values);$('dialog-backdrop').classList.add('hidden')}catch(e){notify(e.message)}};$('dialog-cancel').onclick=()=>popup('dialog-backdrop',false)}
async function importChatgpt(file){if(!file)return;if(file.size>35*1024*1024){notify('For ZIP or large JSON, run: py -3 -m continuum.cli import-chatgpt export.zip');return}if(!file.name.toLowerCase().endsWith('.json')){notify('Use conversations.json here; ZIP imports work through PowerShell');return}let doc;try{doc=JSON.parse(await file.text())}catch{notify('Invalid JSON file');return}const title=prompt('Optional conversation-title filter (blank imports all):','');if(title===null)return;try{const result=await api('/api/import-chatgpt',{document:doc,title_contains:title,redact:true});state.sessionId=null;await refreshProjects(result.project_id);notify(`Imported ${result.conversations} conversations and ${result.chat_messages} messages locally. Credentials redacted by default.`)}catch(e){notify(e.message)}}
async function exportProject(){try{const p=await api('/api/export?'+q({project_id:state.projectId}));download(JSON.stringify(p,null,2),'S-project-'+new Date().toISOString().slice(0,10)+'.json','application/json');notify('Project exported locally')}catch(e){notify(e.message)}}
async function handoff(){try{const data=await api('/api/handoff?'+q({project_id:state.projectId}));download(data.markdown,'S-continuity-'+new Date().toISOString().slice(0,10)+'.md','text/markdown');notify('Continuity capsule saved')}catch(e){notify(e.message)}}
async function exploreRelation(){try{const items=await api('/api/relations?'+q({project_id:state.projectId,word:$('word').value.trim()}));const area=$('relations');area.replaceChildren();if(!items.length){area.textContent='No structural link observed yet.';return}for(const it of items){const article=document.createElement('article');article.className='note';const title=document.createElement('strong');title.textContent=it.word+' · score '+it.score;const body=document.createElement('p');body.textContent=it.reasons.join('; ');article.append(title,body);area.append(article)}}catch(e){notify(e.message)}}
function clearDisplayedApiKey(){
  const secret=$('new-api-key');if(secret)secret.value='';
  const box=$('new-api-key-wrap');if(box)box.classList.add('hidden');
}
async function refreshApiKeys(){
  const list=$('api-key-list');list.replaceChildren();
  try{
    const keys=await api('/api/keys');
    if(!keys.length){list.textContent='No keys created yet.';return}
    for(const key of keys){
      const row=document.createElement('div');row.className='api-key-entry';
      const meta=document.createElement('div');meta.className='key-meta';
      const title=document.createElement('strong');title.textContent=key.name;
      const info=document.createElement('small');
      info.textContent=`${key.prefix}… · ${key.scopes.join(', ')} · ${key.revoked_at?'Revoked':'Active'}`;
      meta.append(title,info);row.append(meta);
      if(!key.revoked_at){
        const btn=document.createElement('button');btn.className='secondary-button';btn.textContent='Revoke';
        btn.onclick=async()=>{
          if(!confirm(`Revoke API key for ${key.name}? Clients using it will lose access.`))return;
          try{await api('/api/revoke-key',{id:key.id});await refreshApiKeys();notify('S API key revoked')}catch(e){notify(e.message)}
        };row.append(btn);
      }
      list.append(row);
    }
  }catch(e){list.textContent='Unable to load keys';notify(e.message)}
}
function createApiKeyDialog(){
  openDialog('Generate a new S API key',[
    {id:'name',label:'App or device name (for example: My Windows Tool)'},
    {id:'scopes',label:'Allowed actions',options:['read,chat','read','chat']}
  ],async values=>{
    const created=await api('/api/keys',{name:values.name,scopes:values.scopes.split(',')});
    clearDisplayedApiKey();
    $('new-api-key').value=created.key;
    $('new-api-key-wrap').classList.remove('hidden');
    await refreshApiKeys();
    notify('Key generated. Copy it now; S will not show it again.');
  });
}
// S Invention Lab: verified symbolic execution is separate from generated chat text.
async function openInventionLab(){switchPanel('invention');await refreshInventions()}
async function refreshInventions(preferred){
 const goals=await api('/api/inventions?'+q({project_id:state.projectId}));
 state.inventionId=(preferred&&goals.some(g=>g.id===preferred))?preferred:
    (goals.some(g=>g.id===state.inventionId)?state.inventionId:(goals[0]?.id||null));
 const list=$('invention-list');list.replaceChildren();
 for(const goal of goals){
   const btn=document.createElement('button');btn.type='button';btn.className='invention-choice';
   btn.textContent=`${goal.objective} — ${goal.status}`;
   if(goal.id===state.inventionId)btn.classList.add('chosen');
   btn.onclick=async()=>{state.inventionId=goal.id;await refreshInventions(goal.id)};
   list.append(btn);
 }
 if(!goals.length){list.textContent='No goals in this project.';$('invention-detail').textContent='Create a goal with observable success criteria.';return}
 await showInventionDetail();
}
async function showInventionDetail(){
 if(!state.inventionId)return;
 const goal=await api('/api/invention?'+q({project_id:state.projectId,goal_id:state.inventionId}));
 const area=$('invention-detail');area.replaceChildren();
 const heading=document.createElement('h3');heading.textContent=goal.objective;
 const criteria=document.createElement('p');criteria.textContent=`Success criteria: ${goal.success_criteria}`;
 const status=document.createElement('p');status.textContent=`Status: ${goal.status}; version: ${goal.version}; candidate: ${goal.candidate?.join(' → ')||'none'}`;
 area.append(heading,criteria,status);
 const buttons=document.createElement('div');buttons.className='lab-controls';
 const addBtn=(name,callback)=>{const b=document.createElement('button');b.type='button';b.className='secondary-button';b.textContent=name;b.onclick=callback;buttons.append(b)};
 addBtn('Teach example',()=>inventionExampleDialog('teach'));
 addBtn('Held-out test',()=>inventionExampleDialog('test'));
 addBtn('Discover rule',()=>runInvention('synthesize'));
 addBtn('Run tests',()=>runInvention('evaluate'));
 addBtn('Predict new input',()=>openDialog('Try learned instruction',[
   {id:'input',label:'New input'}],async v=>{
   const result=await api('/api/invention/predict',{project_id:state.projectId,goal_id:state.inventionId,input:v.input});
   notify(`S predicts: ${result.output} (${result.status})`)
 }));
 area.append(buttons);
 const cases=document.createElement('h4');cases.textContent='Teacher examples and independent tests';area.append(cases);
 for(const ex of goal.examples){
   const row=document.createElement('div');row.className='lab-example';
   const txt=document.createElement('span');txt.textContent=`[${ex.role}] ${ex.input} → ${ex.expected} ${ex.revision?'· corrected '+ex.revision+' time(s)':''}`;
   const edit=document.createElement('button');edit.type='button';edit.className='secondary-button';edit.textContent='Correct';
   edit.onclick=()=>openDialog('Correct the expected result',[
      {id:'expected',label:'Correct result',initial:ex.expected},
      {id:'teacher_note',label:'Why this correction matters',long:true}],async v=>{
      await api('/api/invention/correct',{project_id:state.projectId,goal_id:state.inventionId,
          example_id:ex.id,expected:v.expected,teacher_note:v.teacher_note});
      await refreshInventions(state.inventionId);notify('Correction saved and hypothesis reevaluated')
   });row.append(txt,edit);area.append(row)
 }
 const runs=document.createElement('h4');runs.textContent='Recorded experiments';area.append(runs);
 for(const run of goal.runs){
   const p=document.createElement('div');p.className='lab-run';
   const info=document.createElement('strong');info.textContent=`${run.passed?'PASS':'FAIL'} · version ${run.version} · ${run.evidence_sha256.slice(0,16)}…`;
   const pre=document.createElement('pre');pre.textContent=run.results.map(r=>`${r.input} → ${r.actual===null?'ERROR':r.actual} (expected ${r.expected}) ${r.passed?'PASS':'FAIL'}`).join('\n');
   p.append(info,pre);area.append(p)
 }
}
function inventionExampleDialog(role){
 openDialog(role==='teach'?'Teach S a transformation':'Create a held-out test',[
  {id:'input',label:'Input (e.g. CAT)'},{id:'expected',label:'Expected result (e.g. CATS)'}],async v=>{
    await api('/api/invention/example',{project_id:state.projectId,goal_id:state.inventionId,
       role,input:v.input,expected:v.expected});
    await refreshInventions(state.inventionId)
 });
}
async function runInvention(action){
 try{
  const result=await api(`/api/invention/${action}`,{project_id:state.projectId,goal_id:state.inventionId});
  await refreshInventions(state.inventionId);
  notify(`${result.status}: ${action==='evaluate'?`${result.results.length} held-out case(s) tested`:`${result.consistent_candidates} consistent program(s)`}`)
 }catch(e){notify(e.message)}
}
function newInventionDialog(){
 openDialog('Teach S a new invention',[
   {id:'objective',label:'Goal (observable behavior)',initial:'Learn what adding S does to a symbol'},
   {id:'success_criteria',label:'How should S prove it?',long:true,initial:'Infer a rule from two examples and pass at least one unseen example'}],
 async v=>{const goal=await api('/api/inventions',{project_id:state.projectId,objective:v.objective,success_criteria:v.success_criteria});await refreshInventions(goal.id)})
}

function wire(){setTheme(localStorage.getItem('s-theme')!=='light');$('sidebar-open').onclick=()=>sidebar(!$('sidebar').classList.contains('open'));$('sidebar-close').onclick=()=>sidebar(false);$('mobile-overlay').onclick=()=>sidebar(false);$('model-button').onclick=()=>{const show=$('model-menu').classList.contains('hidden');popup('model-menu',show);$('model-button').setAttribute('aria-expanded',String(show))};document.addEventListener('click',e=>{if(!e.target.closest('.model-control'))popup('model-menu',false);if(!e.target.closest('.sidebar-foot'))popup('profile-menu',false)});$('new-chat').onclick=()=>{showWelcome();switchPanel('chat');sidebar(false);$('chat-input').focus()};$('profile-btn').onclick=()=>popup('profile-menu',$('profile-menu').classList.contains('hidden'));$('theme-toggle').onclick=()=>setTheme(!document.documentElement.classList.contains('dark'));
 $('project').onchange=async e=>{state.projectId=e.target.value;state.sessionId=null;await refreshSessions()};$('new-project').onclick=()=>openDialog('Create a separate S project',[{id:'name',label:'Project name'}],async v=>{const p=await api('/api/projects',{name:v.name});state.sessionId=null;await refreshProjects(p.id)});
 $('composer').onsubmit=e=>{e.preventDefault();send()};$('chat-input').oninput=resizeInput;$('chat-input').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}};$('stop-btn').onclick=()=>{state.controller?.abort();notify('Stopped waiting for local response. An operation already started may finish.')};
 for(const item of document.querySelectorAll('[data-prompt]'))item.onclick=()=>{$('chat-input').value=item.dataset.prompt;resizeInput();$('chat-input').focus()};$('archive-open').onclick=openArchive;$('archive-search').onclick=openArchive;$('knowledge-open').onclick=showKnowledge;$('invention-open').onclick=()=>openInventionLab().catch(e=>notify(e.message));$('new-invention').onclick=newInventionDialog;$('settings-open').onclick=()=>{switchPanel('settings');popup('profile-menu',false);sidebar(false);refreshApiKeys()};for(const btn of document.querySelectorAll('.back-chat'))btn.onclick=()=>switchPanel('chat');$('search-run').onclick=performSearch;$('search-query').onkeydown=e=>{if(e.key==='Enter')performSearch()};$('add-note').onclick=()=>openDialog('Preserve a teaching checkpoint',[{id:'kind',label:'Evidence category',options:['lesson','hypothesis','experiment','evidence','decision','question']},{id:'title',label:'Title'},{id:'body',label:'Content',long:true},{id:'evidence',label:'Evidence reference (required for evidence notes)'}],async v=>{await api('/api/notes',{project_id:state.projectId,...v});await refreshNotes()});$('relate').onclick=exploreRelation;
 $('create-api-key').onclick=createApiKeyDialog;$('refresh-api-keys').onclick=refreshApiKeys;$('copy-api-key').onclick=()=>navigator.clipboard.writeText($('new-api-key').value).then(()=>notify('API key copied')).catch(()=>notify('Clipboard unavailable'));$('hide-api-key').onclick=clearDisplayedApiKey;
 $('export').onclick=exportProject;$('export-settings').onclick=exportProject;$('handoff').onclick=handoff;$('refresh-models').onclick=health;$('import-chatgpt').onclick=()=>$('chatgpt-file').click();$('import-settings').onclick=()=>$('chatgpt-file').click();$('chatgpt-file').onchange=async e=>{await importChatgpt(e.target.files[0]);e.target.value=''};
 $('attach-btn').onclick=()=>$('file-input').click();$('file-input').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{if(f.size>100*1024)throw Error('Text attachment exceeds 100 KiB; paste a relevant excerpt instead');const contents=await f.text();if(contents.includes('\0'))throw Error('Binary attachment is not supported');const suffix=`\n\n[Attached text file: ${f.name}]\n\n${contents.slice(0,25000)}`;$('chat-input').value=($('chat-input').value+suffix).slice(0,49000);$('attachment-info').textContent=`Included ${f.name} as text within the message (${Math.min(contents.length,25000)} characters).`;$('attachment-info').classList.remove('hidden');resizeInput()}catch(e){notify(e.message)}finally{e.target.value=''}};
 $('web-search').onclick=()=>{const input=$('chat-input').value.trim();if(!input){notify('Write a search query in the message box first.');return}if(confirm('Open an external web search for this text? Your query will leave your PC. Search results will not automatically enter S memory.')){window.open('https://www.google.com/search?q='+encodeURIComponent(input),'_blank','noopener,noreferrer');$('search-pill').classList.remove('hidden')}};$('search-pill-close').onclick=()=>popup('search-pill',false);
 $('voice-btn').onclick=()=>notify('Offline speech recognition is not bundled. Windows voice typing: focus the message box and press Win + H. Browser or Windows speech services may use the internet.');resizeInput()}
(async()=>{wire();try{await health();await refreshProjects()}catch(e){notify(e.message)}})();