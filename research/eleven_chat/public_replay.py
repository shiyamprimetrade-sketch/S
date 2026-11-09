"""AXIOM 11-dialogue replay engine (public-safe; no user transcript).

Fixed user text, 10 rule-based alternative answer strategies, and one optional
historical summary baseline. This is an experiment in response policy, not an
LLM generation engine. Private input data stays on the user's computer.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

NAMES = ('Historical summary', 'Procedural', 'Teacher-first', 'First principles',
         'Experimental', 'Build-first', 'Minimal', 'Systems', 'Socratic',
         'Skeptical', 'Adaptive co-inventor')


def reply(style: int, item: dict, previous: list[dict]) -> str:
    core=item.get('core','Interpret the input')
    test=item.get('test','Agree on an observable result')
    alternative=item.get('alternative','Compare alternative approaches')
    goal=item.get('goal','Solve the user request')
    intent=item.get('intent','')
    phase=item.get('phase','')
    if style==0: return item.get('observed_summary','Not available (historical output absent)')
    if style==1: return f'Practical step: {core}. Next: {test}.'
    if style==2: return f'My interpretation: {core}. I do not assume I fully understand. What would you correct in this interpretation?'
    if style==3: return f'Challenge the inherited assumptions: {core}. Alternative design to examine: {alternative}. Test: {test}.'
    if style==4: return f'Testable hypothesis: {alternative}. Experiment: {test}. Keep measured outcomes separate from explanations.'
    if style==5: return f'Implementation requirement: {core}. Deliver a working prototype and verify it by this test: {test}.'
    if style==6: return f'{core}. Next: {test}.'
    if style==7: return f'Goal: {goal}. Input: {item["input"]}. System design: {alternative}. Verify state with: {test}.'
    if style==8: return f'What rule is being questioned? {core}. How would you recognize a successful alternative?'
    if style==9: return f'{core}. Changing a label is different from changing a verified result. {test}.'
    if style==10:
        corrections=[p for p in previous if p.get('intent') in ('feedback','correction')]
        if intent in ('feedback','correction'):
            return f'I am changing my method based on your correction: {core}. I will preserve alternative hypotheses and test the next result: {test}.'
        if intent in ('evidence','report'):
            return f'Observed so far: {core}. The next distinct verification is: {test}. Do not repeat previously verified setup.'
        if phase=='teaching' or corrections:
            return f'I understand the deeper point as {core}. Rather than narrow your idea, consider: {alternative}. We can check it through: {test}.'
        return f'Goal: {goal}. Alternative to build: {alternative}. Proof standard: {test}. Keep the existing checkpoint intact.'
    raise ValueError('unknown policy')


def replay(items: list[dict]) -> dict:
    if not isinstance(items,list) or not items:raise ValueError('expected nonempty list of input items')
    for i,x in enumerate(items):
        if not isinstance(x,dict) or not isinstance(x.get('input'),str) or not x['input']:
            raise ValueError(f'input row {i+1} needs a nonempty input string')
    chats=[]
    for p,name in enumerate(NAMES):
        prior=[];records=[]
        for i,x in enumerate(items,1):
            out=reply(p,x,prior)
            records.append({'turn':i,'input':x['input'],'output':out,'policy':name,
                            'source':'historical summary' if p==0 else 'rule-based counterfactual'})
            prior.append(x)
        chats.append({'id':p,'name':name,'turns':records})
    for i in range(len(items)):
        assert len({ch['turns'][i]['input'] for ch in chats})==1
    return {'method':'deterministic rule-based counterfactual','input_count':len(items),
            'caveat':'The input data supplied determines fidelity. Raw private exports should stay local.',
            'chats':chats}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('private_input_json',type=Path)
    parser.add_argument('--out',type=Path,default=Path('eleven_conversations_private.json'))
    a=parser.parse_args();items=json.loads(a.private_input_json.read_text(encoding='utf8'))
    r=replay(items);a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
    print(f"Saved {len(r['chats'])} paths over {len(items)} fixed inputs to {a.out}")
if __name__=='__main__':main()