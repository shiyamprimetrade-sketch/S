"""ChatGPT data-export importer. No network access and no automatic publishing.

The official export may contain a list of conversations or {'conversations': ...}.
Messages follow the active parent chain of the last node; abandoned edit branches
are not duplicated in a linear discussion. Non-text payload metadata is preserved
as a placeholder; original binary assets are NOT recreated from metadata.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import re
from pathlib import Path

# Avoid copying exposed credentials into searchable text by default.
TOKEN = re.compile(r'(?<![\w-])eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}(?:\.[A-Za-z0-9_-]+)?|(?<![\w-])eyJhIjoi[A-Za-z0-9_-]{60,}|(?i:cloudflared\s+(?:service\s+install|tunnel\s+run\s+--token))\s+[^\s]{30,}')
API_KEY = re.compile(r'(?i)(api[_ -]?key|secret|password|token)\s*[:=]\s*["\']?([A-Za-z0-9_./+-]{28,})')

def redact_secrets(text: str) -> str:
    text = TOKEN.sub('[REDACTED_CREDENTIAL]', text)
    return API_KEY.sub(lambda m: m.group(1)+': [REDACTED_CREDENTIAL]', text)

def timestamp(seconds):
    try:
        return datetime.fromtimestamp(float(seconds),tz=timezone.utc).isoformat(timespec='seconds')
    except (TypeError,ValueError,OverflowError,OSError):
        return datetime.now(timezone.utc).isoformat(timespec='seconds')

def message_text(message):
    """Return readable message text and metadata placeholders without executing rich content."""
    content = (message or {}).get('content') or {}
    parts = content.get('parts') if isinstance(content,dict) else None
    if parts is None:
        parts = [content.get('text','')] if isinstance(content,dict) else [content]
    out=[]
    for part in parts:
        if isinstance(part,str):out.append(part)
        elif isinstance(part,dict):
            if isinstance(part.get('text'),str):out.append(part['text'])
            kind=part.get('content_type',part.get('type','attachment'))
            if part.get('asset_pointer') or kind in {'image_asset_pointer','audio_asset_pointer','video'}:
                out.append(f'[ARCHIVED MEDIA: {kind}; export original separately for binary content]')
            elif not part.get('text'):
                out.append(f'[ARCHIVED NON-TEXT PART: {kind}]')
        elif part is not None:out.append('[ARCHIVED NON-TEXT PART]')
    return '\n'.join(x for x in out if x).strip()

def linear_messages(conv):
    mapping=conv.get('mapping') or {}
    if not isinstance(mapping,dict):return []
    # Prefer the currently visible branch, not every orphaned revision.
    pointer=conv.get('current_node')
    if pointer not in mapping:
        candidates=[(float((n.get('message') or {}).get('create_time') or 0),k) for k,n in mapping.items() if isinstance(n,dict) and n.get('message')]
        pointer=max(candidates,default=(0,None))[1]
    visited=set();chain=[]
    while pointer and pointer in mapping and pointer not in visited:
        visited.add(pointer)
        node=mapping[pointer]
        if isinstance(node,dict) and node.get('message'):
            chain.append(node['message'])
        pointer=node.get('parent') if isinstance(node,dict) else None
    chain.reverse()
    result=[]
    for m in chain:
        author=m.get('author') or {}
        role=author.get('role','unknown')
        text=message_text(m)
        if not text:continue
        result.append({'role':role,'content':text,'created_at':timestamp(m.get('create_time')),
                       'source_id':m.get('id',''), 'model':(m.get('metadata') or {}).get('model_slug','')})
    return result

def select_conversations(document, title_contains=''):
    if isinstance(document,list):convs=document
    elif isinstance(document,dict):convs=document.get('conversations')
    else:convs=None
    if not isinstance(convs,list):raise ValueError('Expected ChatGPT conversations.json (a JSON array)')
    query=title_contains.casefold().strip()
    selected=[]
    for c in convs:
        if not isinstance(c,dict):continue
        title=str(c.get('title') or 'Untitled conversation')[:120]
        if query and query not in title.casefold():continue
        selected.append((title,c))
    return selected

def parse_export(document, title_contains='', redact=True, max_messages=100000):
    selected=select_conversations(document,title_contains)
    if not selected:raise ValueError('No matching conversations; check the title filter')
    out=[];total=0
    for title,c in selected:
        msgs=linear_messages(c)
        if total+len(msgs)>max_messages:raise ValueError('Export exceeds the local safety limit; import one title at a time')
        total+=len(msgs)
        if redact:
            msgs=[{**m,'content':redact_secrets(m['content'])} for m in msgs]
        out.append({'title':title,'source_id':c.get('id',''),'messages':msgs})
    return out

def load_export(path):
    path=Path(path)
    if path.stat().st_size>300*1024*1024:raise ValueError('Archive exceeds 300 MiB')
    if path.suffix.lower()=='.zip':
        from zipfile import ZipFile
        with ZipFile(path) as z:
            names=[n for n in z.namelist() if Path(n).name=='conversations.json']
            if not names:raise ValueError('No conversations.json in ZIP')
            info=z.getinfo(names[0])
            if info.file_size>200*1024*1024:raise ValueError('Conversation JSON too large')
            with z.open(info) as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf8'))
