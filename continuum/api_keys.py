"""Cryptographically random, scoped S-issued API keys; SQLite stores only SHA-256."""
from __future__ import annotations
import hashlib
import hmac
import json
import re
import secrets
from .storage import now

PERMISSIONS = frozenset({'read','chat'})
KEY_PATTERN = re.compile(r'^sk_s_([a-f0-9]{16})_([A-Za-z0-9_-]{43})$')

def _permissions(scopes):
    if isinstance(scopes,str):scopes=scopes.split(',')
    if not isinstance(scopes,(tuple,list,set,frozenset)):raise ValueError('Scopes must be read, chat, or both')
    clean=tuple(sorted({str(scope).strip() for scope in scopes}))
    if not clean or any(scope not in PERMISSIONS for scope in clean):
        raise ValueError('Scopes must be read, chat, or both')
    return list(clean)

def create_key(store,name,scopes=('read','chat')):
    if not isinstance(name,str) or not 1<=len(name.strip())<=80:
        raise ValueError('Key name must contain 1 to 80 characters')
    scopes=_permissions(scopes)
    key_id=secrets.token_hex(8)
    token=f'sk_s_{key_id}_{secrets.token_urlsafe(32)}'
    digest=hashlib.sha256(token.encode('ascii')).hexdigest()
    created_at=now()
    with store.connect() as db:
        db.execute('INSERT INTO api_keys (id,name,key_hash,scopes,created_at) VALUES (?,?,?,?,?)',
                   (key_id,name.strip(),digest,json.dumps(scopes),created_at))
    return {'id':key_id,'name':name.strip(),'prefix':f'sk_s_{key_id}_',
            'scopes':scopes,'created_at':created_at,'key':token}

def list_keys(store):
    with store.connect() as db:
        rows=db.execute('SELECT id,name,scopes,created_at,last_used_at,revoked_at FROM api_keys ORDER BY created_at DESC,id DESC').fetchall()
    return [{'id':r['id'],'name':r['name'],'prefix':f"sk_s_{r['id']}_",
             'scopes':json.loads(r['scopes']),'created_at':r['created_at'],
             'last_used_at':r['last_used_at'],'revoked_at':r['revoked_at']} for r in rows]

def revoke_key(store,key_id):
    if not isinstance(key_id,str) or not re.fullmatch('[a-f0-9]{16}',key_id):
        raise ValueError('Invalid key ID')
    with store.connect() as db:
        changed=db.execute('UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL',
                           (now(),key_id)).rowcount
    return {'id':key_id,'revoked':bool(changed)}

def authenticate_key(store,token,scope):
    if scope not in PERMISSIONS or not isinstance(token,str):return False
    match=KEY_PATTERN.fullmatch(token)
    if not match:return False
    key_id=match.group(1)
    digest=hashlib.sha256(token.encode('ascii')).hexdigest()
    with store.connect() as db:
        row=db.execute('SELECT key_hash,scopes,revoked_at FROM api_keys WHERE id=?',(key_id,)).fetchone()
        if row is None:return False
        matches=hmac.compare_digest(digest,row['key_hash'])
        if not matches or row['revoked_at'] is not None or scope not in json.loads(row['scopes']):
            return False
        db.execute('UPDATE api_keys SET last_used_at=? WHERE id=?',(now(),key_id))
    return True
