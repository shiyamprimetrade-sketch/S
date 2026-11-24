"""Bounded Unity v3 orchestration over trusted pinned Alpha/Omega modules."""
import hashlib
import hmac
import json
from .local_transport import canonical, execute_local

def run(task, signature, key, root, db, journal_db, *, allowed, verifiers):
    from journal import Journal
    from transactional import seal, verify_record
    raw = canonical(task)
    snapshot = json.loads(raw)
    if not isinstance(key, bytes) or len(key) < 32 or len(raw)>65536:
        raise ValueError("invalid credentials or size")
    expected = hmac.new(key,b"alpha-fusion-v3\0"+raw,hashlib.sha256).hexdigest()
    if not isinstance(signature,str) or not signature.isascii() or not hmac.compare_digest(signature,expected):
        raise ValueError("authentication failed")
    if not isinstance(snapshot,dict) or set(snapshot)!={"version","id","operation","args"}:
        raise ValueError("invalid envelope")
    import re
    ident, op = snapshot["id"], snapshot["operation"]
    if (type(snapshot["version"]) is not int or snapshot["version"]!=3 or
        not isinstance(ident,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",ident) or
        not isinstance(op,str) or not isinstance(snapshot["args"],dict) or
        op not in allowed or op not in verifiers):
        raise ValueError("operation denied or no independent verifier")
    journal=Journal(str(journal_db))
    journal.begin(ident)
    record={"id":ident,"request_digest":hashlib.sha256(raw).hexdigest()}
    try:
        evidence=execute_local(snapshot,signature,key,root,db,allowed=allowed)
        result=evidence["record"]["result"]
        if result["status"]!="ok" or verifiers[op](snapshot,result,root,db) is not True:
            raise ValueError("independent acceptance failed")
        record.update(status="completed",result_digest=hashlib.sha256(canonical(result)).hexdigest())
    except Exception as exc:
        record.update(status="failed",error=type(exc).__name__)
        journal.finish(ident,"failed",seal(record,key))
        raise
    sealed=seal(record,key)
    journal.finish(ident,"completed",sealed)
    if not verify_record(ident,"completed",sealed,key) or journal.get(ident)!=("completed",sealed):
        raise RuntimeError("journal verification failed")
    return result,sealed

def pending(journal_db):
    """Report interrupted candidates; never automatically replay side effects."""
    from journal import Journal
    return Journal(str(journal_db)).pending()
