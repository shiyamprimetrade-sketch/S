"""S-Memory: local, permissioned, source-attributed SQLite retrieval.

Memory is a separate subsystem from neural network weights. Remembering a
sentence does not mean S learned it through backpropagation. Records can be
retrieved and supplied to a future prompt only with explicit scope consent.
No automatic capture of conversations, network access or user profiling.

SQLite transactions protect updates, foreign scopes are never searched,
and expiry and deletion are enforced before retrieval. Queries use lexical
BM25-style term statistics in Python so builds do not depend on SQLite FTS5.
Raw content remains in an operator-chosen local database, not in GitHub.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Iterable

WORDS=re.compile(r"\w+",re.UNICODE)
SENSITIVE=re.compile(
    r"(?i)(?:password|secret|access[_-]?token|api[_-]?key)\s*[:=]\s*\S+"
)
EMAIL=re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}",re.I)


@dataclasses.dataclass(frozen=True)
class MemoryPolicy:
    maximum_items: int = 2000
    maximum_content_bytes: int = 8192
    maximum_query_bytes: int = 2048
    maximum_results: int = 20
    minimum_ttl_seconds: int = 1
    maximum_ttl_seconds: int = 365 * 86400
    prohibit_secret_patterns: bool = True
    prohibit_email_patterns: bool = True

    def validate(self):
        if min(self.maximum_items,self.maximum_content_bytes,
               self.maximum_query_bytes,self.maximum_results,
               self.minimum_ttl_seconds)>0 is False:
            raise ValueError("Memory bounds must be positive")
        if self.maximum_ttl_seconds < self.minimum_ttl_seconds:
            raise ValueError("Memory TTL bounds inverted")


@dataclasses.dataclass(frozen=True)
class MemoryRecord:
    id: str
    scope: str
    content: str
    source: str
    source_sha256: str
    consent: str
    created_at: int
    expires_at: int
    revision: int
    content_sha256: str
    tags: tuple[str, ...] = ()

    def redacted_metadata(self):
        return {
            "id":self.id,"scope":self.scope,"source":self.source,
            "source_sha256":self.source_sha256,
            "created_at":self.created_at,"expires_at":self.expires_at,
            "revision":self.revision,"content_sha256":self.content_sha256,
            "tags":list(self.tags),
        }


@dataclasses.dataclass(frozen=True)
class RetrievalHit:
    record: MemoryRecord
    score: float
    matched_terms: tuple[str, ...]

    def evidence(self):
        return {
            "memory_id":self.record.id,
            "source":self.record.source,
            "source_sha256":self.record.source_sha256,
            "content_sha256":self.record.content_sha256,
            "retrieval_score":self.score,
            "matched_terms":list(self.matched_terms),
        }


def content_hash(content:str)->str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def tokens(text:str)->list[str]:
    return [x.casefold() for x in WORDS.findall(text)]


def validate_scope(scope:str):
    if not isinstance(scope,str) or not 1<=len(scope)<=128:
        raise ValueError("Scope must be 1..128 characters")
    if not re.fullmatch(r"[a-zA-Z0-9._:-]+",scope):
        raise ValueError("Unsafe project scope")


def validate_source(source:str):
    if not isinstance(source,str) or not 1<=len(source)<=512:
        raise ValueError("Named memory source required")


def validate_consent(consent:str):
    if consent not in ("explicit", "import-authorized"):
        raise PermissionError("Explicit permission required to persist memory")


def evaluate_content(content:str,policy:MemoryPolicy):
    if not isinstance(content,str) or not content.strip():
        raise ValueError("Memory content must be nonempty")
    if len(content.encode("utf-8"))>policy.maximum_content_bytes:
        raise ValueError("Memory exceeds byte limit")
    if policy.prohibit_secret_patterns and SENSITIVE.search(content):
        raise ValueError("Potential credential found; refusing storage")
    if policy.prohibit_email_patterns and EMAIL.search(content):
        raise ValueError("Potential email found; refusing storage")


def bm25(
    query_terms:list[str],document_tokens:list[str],
    document_frequency:dict[str,int],total_docs:int,
    average_length:float,k1:float=1.5,b:float=0.75,
)->float:
    if not document_tokens or not query_terms or total_docs<1:
        return 0.0
    counts=collections.Counter(document_tokens)
    length=len(document_tokens)
    score=0.0
    for term in set(query_terms):
        tf=counts.get(term,0)
        if not tf:
            continue
        df=document_frequency.get(term,0)
        idf=math.log(1+(total_docs-df+0.5)/(df+0.5))
        scale=tf+k1*(1-b+b*length/max(1,average_length))
        score+=idf*(tf*(k1+1)/scale)
    return score


def rank_records(query:str,rows:list[MemoryRecord],
                 limit:int=5)->list[RetrievalHit]:
    if limit<1:
        raise ValueError("Result limit must be positive")
    query_terms=tokens(query)
    if not query_terms:
        return []
    vectors=[tokens(row.content) for row in rows]
    frequency=collections.Counter()
    for document in vectors:
        frequency.update(set(document))
    average=sum(map(len,vectors))/max(1,len(vectors))
    ranked=[]
    for row,vector in zip(rows,vectors):
        score=bm25(query_terms,vector,frequency,len(rows),average)
        if score>0:
            matched=tuple(sorted(set(query_terms)&set(vector)))
            ranked.append(RetrievalHit(row,score,matched))
    ranked.sort(key=lambda x:(-x.score,x.record.created_at,x.record.id))
    return ranked[:limit]


class MemoryStore:
    """Explicitly opened SQLite database with per-project access boundaries."""

    def __init__(self,path:Path,*,policy:MemoryPolicy|None=None,
                 clock:Callable[[],float]=time.time):
        self.path=Path(path)
        self.policy=policy or MemoryPolicy()
        self.policy.validate()
        self.clock=clock
        self.lock=threading.RLock()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(str(path),check_same_thread=False)
        self.db.row_factory=sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=10000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories(
                id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                consent TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                tags_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(scope,expires_at);
            CREATE TABLE IF NOT EXISTS memory_audit(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                time_at INTEGER NOT NULL,
                operation TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                content_sha256 TEXT NOT NULL
            );
        """)
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()

    def _row(self,r):
        return MemoryRecord(
            id=r["id"],scope=r["scope"],content=r["content"],
            source=r["source"],source_sha256=r["source_sha256"],
            consent=r["consent"],created_at=r["created_at"],
            expires_at=r["expires_at"],revision=r["revision"],
            content_sha256=r["content_sha256"],
            tags=tuple(json.loads(r["tags_json"])),
        )

    def _audit(self,operation:str,record:MemoryRecord):
        self.db.execute(
            "INSERT INTO memory_audit(time_at,operation,memory_id,scope,content_sha256) VALUES(?,?,?,?,?)",
            (int(self.clock()),operation,record.id,record.scope,record.content_sha256),
        )

    def expire(self,scope:str|None=None)->int:
        if scope is not None:
            validate_scope(scope)
        with self.lock,self.db:
            now=int(self.clock())
            if scope is None:
                expired=self.db.execute("SELECT * FROM memories WHERE expires_at<=?",(now,)).fetchall()
            else:
                expired=self.db.execute(
                    "SELECT * FROM memories WHERE scope=? AND expires_at<=?",
                    (scope,now),
                ).fetchall()
            for row in expired:
                record=self._row(row)
                self._audit("expire",record)
                self.db.execute("DELETE FROM memories WHERE id=?",(record.id,))
            return len(expired)

    def count(self,scope:str)->int:
        validate_scope(scope)
        self.expire(scope)
        with self.lock:
            return self.db.execute(
                "SELECT count(*) FROM memories WHERE scope=?",(scope,)
            ).fetchone()[0]

    def remember(self,*,scope:str,content:str,source:str,
                 consent:str,ttl_seconds:int=86400,
                 tags:Iterable[str]=())->MemoryRecord:
        validate_scope(scope)
        validate_source(source)
        validate_consent(consent)
        evaluate_content(content,self.policy)
        if not self.policy.minimum_ttl_seconds<=ttl_seconds<=self.policy.maximum_ttl_seconds:
            raise ValueError("TTL outside policy")
        clean_tags=tuple(sorted(set(tags)))
        if any(not re.fullmatch(r"[A-Za-z0-9._-]{1,32}",tag) for tag in clean_tags):
            raise ValueError("Invalid memory tag")
        now=int(self.clock())
        record=MemoryRecord(
            id=str(uuid.uuid4()),scope=scope,content=content,source=source,
            source_sha256=content_hash(source),consent=consent,
            created_at=now,expires_at=now+ttl_seconds,revision=1,
            content_sha256=content_hash(content),tags=clean_tags,
        )
        with self.lock,self.db:
            self.expire(scope)
            existing=self.db.execute(
                "SELECT count(*) FROM memories WHERE scope=?",(scope,)
            ).fetchone()[0]
            if existing>=self.policy.maximum_items:
                raise ValueError("Scope memory quota exceeded")
            self.db.execute(
                "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (record.id,record.scope,record.content,record.source,
                 record.source_sha256,record.consent,record.created_at,
                 record.expires_at,record.revision,record.content_sha256,
                 json.dumps(record.tags)),
            )
            self._audit("remember",record)
        return record

    def list_scope(self,scope:str)->list[MemoryRecord]:
        validate_scope(scope)
        self.expire(scope)
        with self.lock:
            rows=self.db.execute(
                "SELECT * FROM memories WHERE scope=? ORDER BY created_at,id",(scope,)
            ).fetchall()
            return [self._row(row) for row in rows]

    def get(self,scope:str,memory_id:str)->MemoryRecord|None:
        validate_scope(scope)
        self.expire(scope)
        with self.lock:
            row=self.db.execute(
                "SELECT * FROM memories WHERE scope=? AND id=?",
                (scope,memory_id),
            ).fetchone()
            return self._row(row) if row else None

    def retrieve(self,scope:str,query:str,limit:int=5)->list[RetrievalHit]:
        validate_scope(scope)
        if not isinstance(query,str) or len(query.encode("utf-8"))>self.policy.maximum_query_bytes:
            raise ValueError("Invalid query size")
        if not 1<=limit<=self.policy.maximum_results:
            raise ValueError("Result limit outside policy")
        return rank_records(query,self.list_scope(scope),limit)

    def update(self,scope:str,memory_id:str,content:str,
               *,consent:str)->MemoryRecord:
        validate_scope(scope)
        validate_consent(consent)
        evaluate_content(content,self.policy)
        with self.lock,self.db:
            old=self.get(scope,memory_id)
            if old is None:
                raise KeyError("Memory absent in this scope")
            current=dataclasses.replace(
                old,content=content,content_sha256=content_hash(content),
                revision=old.revision+1,consent=consent,
            )
            self.db.execute(
                "UPDATE memories SET content=?,content_sha256=?,revision=?,consent=? WHERE scope=? AND id=?",
                (current.content,current.content_sha256,current.revision,
                 current.consent,scope,memory_id),
            )
            self._audit("update",current)
            return current

    def forget(self,scope:str,memory_id:str)->bool:
        validate_scope(scope)
        with self.lock,self.db:
            record=self.get(scope,memory_id)
            if record is None:
                return False
            self._audit("forget",record)
            self.db.execute(
                "DELETE FROM memories WHERE scope=? AND id=?",(scope,memory_id)
            )
            return True

    def forget_scope(self,scope:str)->int:
        validate_scope(scope)
        count=0
        for row in self.list_scope(scope):
            count+=int(self.forget(scope,row.id))
        return count

    def audit(self,scope:str)->list[dict]:
        validate_scope(scope)
        with self.lock:
            rows=self.db.execute(
                "SELECT sequence,time_at,operation,memory_id,content_sha256 FROM memory_audit WHERE scope=? ORDER BY sequence",
                (scope,),
            ).fetchall()
            return [dict(row) for row in rows]

    def export_metadata(self,scope:str)->dict:
        rows=self.list_scope(scope)
        return {
            "scope":scope,"count":len(rows),
            "memories":[row.redacted_metadata() for row in rows],
            "raw_content_excluded":True,
        }


def citation_context(hits:list[RetrievalHit],max_bytes:int=4096)->str:
    """Treat retrieved memory as untrusted, quoted evidence, not instructions."""
    if max_bytes<1:
        raise ValueError("Context budget must be positive")
    lines=[]
    remaining=max_bytes
    for hit in hits:
        content=hit.record.content.replace("\x00","")
        item=(
            f'[memory_id={hit.record.id} source={hit.record.source_sha256}]\n'
            f'{content}\n[/memory]\n'
        )
        size=len(item.encode("utf-8"))
        if size>remaining:
            continue
        lines.append(item)
        remaining-=size
    return "".join(lines)


def memory_quality(relevant_ids:set[str],hits:list[RetrievalHit],k:int)->dict:
    if k<1:
        raise ValueError("k must be positive")
    selected={hit.record.id for hit in hits[:k]}
    true=len(selected&relevant_ids)
    return {
        "precision_at_k":true/max(1,min(k,len(hits))),
        "recall_at_k":true/max(1,len(relevant_ids)),
        "hits_at_k":true,
        "requested_k":k,
    }
