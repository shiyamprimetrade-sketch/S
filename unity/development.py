"""Explicitly authorized development workflow; no arbitrary code execution."""
import hashlib
import hmac
import json
from .local_transport import canonical
from .orchestrator import run

def develop(requirement, *, planner, authorize, key, root, db, journal_db,
            allowed, verifiers, repair=None, max_repairs=1):
    if not isinstance(requirement,str) or not requirement.strip() or len(requirement)>16000:
        raise ValueError("invalid requirement")
    if type(max_repairs) is not int or not 0<=max_repairs<=2:
        raise ValueError("invalid repair budget")
    task=planner(requirement)
    attempts=[]
    seen=set()
    for index in range(max_repairs+1):
        raw=canonical(task)
        task=json.loads(raw)
        if not isinstance(task,dict) or set(task)!={"version","id","operation","args"}:
            raise ValueError("invalid plan")
        ident=task["id"]
        if not isinstance(ident,str) or ident in seen:
            raise ValueError("repair must have fresh task identity")
        seen.add(ident)
        if authorize(requirement,json.loads(raw)) is not True:
            return {"status":"denied","attempts":attempts}
        signature=hmac.new(key,b"alpha-fusion-v3\0"+raw,hashlib.sha256).hexdigest()
        try:
            result,evidence=run(task,signature,key,root,db,journal_db,
                                allowed=allowed,verifiers=verifiers)
        except Exception as exc:
            attempts.append({"id":ident,"status":"failed","error":type(exc).__name__})
            # Never automatically repeat a potentially mutating operation.
            if task["operation"]!="calculator" or repair is None or index==max_repairs:
                return {"status":"failed","attempts":attempts}
            task=repair(requirement,json.loads(raw),type(exc).__name__)
            continue
        attempts.append({"id":ident,"status":"verified"})
        return {"status":"delivered","attempts":attempts,"result":result,"evidence":evidence}
    raise RuntimeError("unreachable")
