'use strict';
let codingArtifact=null,codingPath=null,codingDirty=false;
hasSourceEdits=()=>codingDirty;
onProjectChange=()=>{
  codingArtifact=null;codingPath=null;codingDirty=false;
  $('code-files').replaceChildren();$('code-history').replaceChildren();$('code-content').value='';
  $('code-revision').textContent='No source checkpoint loaded for this saved specification.';
  $('code-output').textContent='No build loaded for this project.';$('code-consent').checked=false;
};
function codingProject() {
  if(!current)throw Error('Save and approve a project specification first.');
  if(editVersion)throw Error('Save and approve the changed requirements before generating or building.');
  if(current.status!=='approved')throw Error('Approve the complete saved requirements first.');
  return current.project_id;
}
function codingConsent() {if(!$('code-consent').checked)throw Error('Explicit action approval is required.');}
function codingRender(artifact) {
  if(!current || artifact.project_id!==current.project_id || artifact.spec_revision!==current.revision)
    throw Error('Discarded source from a different project or requirements revision.');
  codingArtifact=artifact;codingDirty=false;codingPath=Object.keys(artifact.files)[0];
  $('code-files').replaceChildren();
  for(const path of Object.keys(artifact.files)) {
    const button=node('button',path);button.type='button';button.onclick=()=>{
      if(busy)return;
      if(codingDirty&&!window.confirm('Discard unsaved edits and select another file?'))return;
      codingPath=path;codingDirty=false;$('code-content').value=artifact.files[path];$('code-filename').textContent=path;
    };$('code-files').append(button);
  }
  $('code-content').value=artifact.files[codingPath];$('code-filename').textContent=codingPath;
  $('code-revision').textContent=`Source revision ${artifact.revision} · ${artifact.provenance.provider} · unverified until built`;
  $('code-recipe').value=artifact.recipe;
}
async function codingHistory() {
  const pid=codingProject();const list=await api('artifacts?project_id='+encodeURIComponent(pid));$('code-history').replaceChildren();
  for(const item of list) {
    const button=node('button',`Restore source ${item.revision}`);button.type='button';button.onclick=()=>action(async()=>{
      codingProject();codingConsent();
      if(!window.confirm('Restore this saved source as a new unverified checkpoint? Unsaved edits will be discarded.'))return;
      const result=await api('restore-source',{project_id:pid,revision:item.revision,
        expected_artifact_revision:codingArtifact.revision,consent:true});
      codingRender(result);await codingHistory();$('code-consent').checked=false;
    });$('code-history').append(button);
  }
}
async function codingLoad() {
  const pid=codingProject();
  if(codingDirty&&!window.confirm('Discard unsaved source edits and reload?'))return;
  const artifact=await api('artifact?project_id='+encodeURIComponent(pid));codingRender(artifact);await codingHistory();
  const runs=await api('runs?project_id='+encodeURIComponent(pid));$('code-output').textContent=JSON.stringify(runs,null,2);
}
$('code-open').onclick=()=>action(async()=>{
  $('code-panel').hidden=false;
  const capabilities=await api('capabilities');$('code-capabilities').textContent=
    `S Native: ${capabilities.providers.native?'checkpoint configured':'not installed'}. Runtime: ${capabilities.native_backend}. `+
    `Container builds: ${capabilities.execution_enabled?'enabled by operator':'disabled by operator'}. `+
    `External local model: ${capabilities.providers['ollama-external']?'configured':'not configured'}.`;
  if(current&&current.status==='approved') {
    const rows=await api('artifacts?project_id='+encodeURIComponent(codingProject()));
    if(rows.length)await codingLoad();
  }
});
$('code-refresh').onclick=()=>action(codingLoad);
$('neural-probe').onclick=()=>action(async()=>{
  if(!current)throw Error('Save a project draft before recording a runtime test.');
  codingConsent();
  const pid=current.project_id;
  const result=await api('neural-probe',{project_id:pid,prompt:$('neural-prompt').value,tokens:16,consent:true});
  if(!current||current.project_id!==pid)throw Error('Discarded a runtime result from another project.');
  $('code-output').textContent=JSON.stringify(result,null,2);$('code-consent').checked=false;
  message('Neural calculations completed. This is a runtime check; coding ability is still unverified.');
});
$('evaluate-coding').onclick=()=>action(async()=>{
  if(!current)throw Error('Save a project draft to record evaluation evidence.');
  codingConsent();const pid=current.project_id;
  const execute=$('evaluation-execute').checked;
  if(execute&&!window.confirm('Execute synthetic evaluation code in the prepared isolated containers?'))return;
  $('code-output').textContent='Running three bounded coding diagnostics. Provider generation and independent execution are assessed separately…';
  const result=await api('evaluate-coding',{project_id:pid,provider:$('code-provider').value,
    execute,max_repairs:0,consent:true});
  if(!current||current.project_id!==pid)throw Error('Discarded evaluation from another project.');
  $('code-output').textContent=JSON.stringify(result,null,2);$('code-consent').checked=false;
  const assessed=result.tasks_fully_assessed||0;
  message((assessed?`Coding diagnostic: ${result.final_tasks_correct}/${assessed} assessed tasks correct. `:
    `Source format: ${result.format_passed||0}/${result.task_count}. Independent execution was not assessed. `)+
    'General capability is not established.');
});
$('code-generate').onclick=()=>action(async()=>{
  const pid=codingProject();codingConsent();
  if(codingDirty&&!window.confirm('Discard unsaved source edits and create a new generated checkpoint?'))return;
  const rows=await api('artifacts?project_id='+encodeURIComponent(pid));
  const result=await api('generate',{project_id:pid,spec_revision:current.revision,
    expected_artifact_revision:rows.length?rows[0].revision:0,recipe:$('code-recipe').value,
    provider:$('code-provider').value,consent:true});
  codingRender(result);await codingHistory();$('code-consent').checked=false;
  message('Source created. Review provenance and run independent checks before treating it as working software.');
});
$('code-content').oninput=()=>{codingDirty=true;message('Unsaved source changes. Save a source checkpoint before building.');};
$('code-save').onclick=()=>action(async()=>{
  const pid=codingProject();codingConsent();if(!codingArtifact||!codingPath)throw Error('Load source first.');
  const result=await api('edit',{project_id:pid,expected_artifact_revision:codingArtifact.revision,
    path:codingPath,content:$('code-content').value,consent:true});
  codingRender(result);await codingHistory();$('code-consent').checked=false;message('Source checkpoint saved.');
});
$('code-build').onclick=()=>action(async()=>{
  const pid=codingProject();codingConsent();if(!codingArtifact)throw Error('Generate source first.');
  if(codingDirty)throw Error('Save source changes before execution.');
  const maximum=Number($('code-repairs').value);
  if(!Number.isInteger(maximum)||maximum<0||maximum>2)throw Error('Repair budget must be 0–2.');
  if(!window.confirm(`Run this source in the configured isolated container, allowing at most ${maximum} repair(s)?`))return;
  $('code-output').textContent='Build and test running in the configured container…';
  const result=await api('build',{project_id:pid,artifact_revision:codingArtifact.revision,
    repair_provider:$('code-provider').value,max_repairs:maximum,consent:true});
  codingRender(result.artifact);await codingHistory();$('code-consent').checked=false;
  $('code-output').textContent=JSON.stringify(result,null,2);message('Build outcome: '+result.final_status+'. Business acceptance still requires independent verification.');
});
$('code-export').onclick=()=>action(async()=>{
  const pid=codingProject();const result=await api('archive?project_id='+encodeURIComponent(pid));
  const bytes=Uint8Array.from(atob(result.archive_base64),c=>c.charCodeAt(0));
  const url=URL.createObjectURL(new Blob([bytes],{type:'application/zip'}));
  const link=node('a','');link.href=url;link.download=result.filename;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);message('Downloaded the saved source checkpoint with provenance.');
});
