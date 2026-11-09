"""Explainable structural word connections. Not semantic intelligence."""
import re
from collections import Counter
from itertools import combinations

def words(text):
    return re.findall(r"[A-Z]{3,24}",text.upper())[:180]

def learn_words(store,pid,text):
    raw=words(text)
    seen=list(dict.fromkeys(raw))[:45]
    if not seen:return
    counts=Counter(raw)
    with store.connect() as db:
        for w,n in counts.items():
            db.execute('INSERT INTO word_counts VALUES(?,?,?) ON CONFLICT(project_id,token) DO UPDATE SET count=count+excluded.count',(pid,w,n))
        for a,b in combinations(sorted(seen),2):
            db.execute('INSERT INTO word_links VALUES(?,?,?,1) ON CONFLICT(project_id,a,b) DO UPDATE SET count=count+1',(pid,a,b))

def related(store,pid,token,limit=20):
    token=token.upper().strip()
    if not re.fullmatch(r'[A-Z]{3,24}',token):raise ValueError('Use a word with 3–24 letters')
    store.get_project(pid)
    with store.connect() as db:
        rec=db.execute('SELECT token,count FROM word_counts WHERE project_id=?',(pid,)).fetchall()
        linkrows=db.execute('SELECT a,b,count FROM word_links WHERE project_id=? AND (a=? OR b=?) ORDER BY count DESC LIMIT 200',(pid,token,token)).fetchall()
    counts={r['token']:r['count'] for r in rec}
    links={r['b'] if r['a']==token else r['a']:r['count'] for r in linkrows}
    results=[]
    for candidate,cnt in counts.items():
        if candidate==token:continue
        equal_len=len(candidate)==len(token)
        anagram=sorted(candidate)==sorted(token)
        together=links.get(candidate,0)
        if not (equal_len or anagram or together):continue
        score=0.2*(equal_len)+1.2*(anagram)+min(2.0,0.3*together)
        reasons=[]
        if equal_len:reasons.append(f'both have {len(token)} letters')
        if anagram:reasons.append('same letter counts in a different order')
        if together:reasons.append(f'co-occurred in {together} message(s)')
        results.append({'word':candidate,'score':round(score,2),'reasons':reasons,'observed_count':cnt})
    return sorted(results,key=lambda r:(-r['score'],r['word']))[:max(1,min(100,limit))]
