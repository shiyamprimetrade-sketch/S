"""S Research: independently runnable, rule-based research assistant.

This is NOT a generative LLM. It decomposes an objective, retrieves explicitly
stored lessons, and proposes falsifiable tests. It makes no network requests.
"""
from __future__ import annotations
import re
from collections import Counter

STOP = {'this','that','what','when','where','which','with','from','into','your','you','our','they','them','their','then','will','have','been','were','does','could','should','would','the','and','for','are','can','how','why','not','all','about','like','make','want','please','just','now','there','only','more','than','without','something','everything','using','computer','system','project','think'}
RULES={
    'invention': {'invent','build','create','design','implement','reprogram','develop','engineer','prototype'},
    'explanation': {'why','what','how','meaning','explain','define','understand','language','symbol'},
    'verification': {'verify','test','proof','evidence','accurate','success','error','correct','measure'},
    'continuity': {'remember','memory','continue','restore','import','backup','subscription','history'},
}

def tokens(text):
    return [w for w in re.findall(r"[^\W_]{3,}",str(text).casefold()) if w not in STOP]

def respond(store,project_id,prompt):
    """Generate an evidence-labeled scaffold without claiming model understanding."""
    prompt=str(prompt).strip()
    if not prompt:
        raise ValueError('Empty request')
    words=tokens(prompt)
    raw=re.findall(r"[^\W_]{2,}", prompt.casefold())
    signals=set(raw) | {w[:-1] for w in raw if w.endswith('s') and len(w)>4}
    labels=[(name, len(signals.intersection(terms))) for name,terms in RULES.items()]
    labels.sort(key=lambda x:x[1],reverse=True)
    intent=labels[0][0] if labels and labels[0][1] else 'exploration'
    notes=store.list_notes(project_id,100)
    scored=[]
    for note in notes:
        q=set(words)
        n=set(tokens(note['title']+' '+note['body']))
        score=len(q.intersection(n))
        if score:scored.append((score,note))
    scored.sort(key=lambda x:(-x[0],x[1]['title']))
    matches=[n for _,n in scored[:3]]
    # Ask a meaningful, content-specific question; avoid copying a generic answer.
    output=["S RESEARCH · Local, rule-based analysis (not an LLM)",
            "",f"Objective received: {prompt[:1000]}",
            f"Interpreted direction: {intent} (keyword-based hypothesis, not verified understanding).",
            "", "Existing knowledge relevant to this objective:"]
    if matches:
        for n in matches:
            output.append(f"• [{n['kind'].upper()}] {n['title']}: {n['body'][:350]}")
    else:
        output.append('• No matching verified project note. I will not invent prior evidence.')
    output.extend(['','Proposed next operation:'])
    if intent=='invention':
        output.append('1. State the observable result this invention must produce.\n2. Build the smallest end-to-end implementation that demonstrates it.\n3. Compare it against the existing method and save the test evidence.')
    elif intent=='verification':
        output.append('1. Define the expected behavior and postcondition.\n2. Run a reproducible test with a negative control.\n3. Record actual output and classify pass/fail from evidence.')
    elif intent=='continuity':
        output.append('1. Check the current local database and last verified checkpoint.\n2. Save a validated backup and a human-readable handoff capsule.\n3. Restore on a second machine and compare message counts and hashes.')
    elif intent=='explanation':
        output.append('1. Separate symbols, human-assigned names, observable behavior and physical mechanism.\n2. Identify which conventions can be changed.\n3. Propose an experiment that distinguishes a re-labeling from a new capability.')
    else:
        output.append('1. Inspect the goal and its context.\n2. Generate at least two explanations or methods.\n3. Test one distinguishing prediction, preserve the observations, then revise.')
    output.extend(['','Status: This is a research proposal, not an executed task or verified new invention.',
                   'Teach me the observed result or save evidence in Knowledge to update the project.'])
    return '\n'.join(output)
