"""Manual 00 versioned boundary contracts; validation is not authorization."""
import re
import uuid
import json

CAPABILITIES = frozenset(("calculator", "website", "inventory"))
STATES = frozenset(("CREATED","UNDERSTANDING","PLANNING","AWAITING_AUTHORIZATION","AUTHORIZED","GENERATING","EXECUTING","VERIFYING","REPAIRING","COMPLETED","DENIED","FAILED","BLOCKED","RECOVERING"))
def _fields(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("Invalid contract fields")
def _id(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Invalid operation identifier")
def validate(request):
    _fields(request, ("schema_version","operation_id","capability","deadline_ms","payload"))
    if type(request["schema_version"]) is not int or request["schema_version"] != 1:
        raise ValueError("Unsupported schema")
    _id(request["operation_id"])
    if type(request["capability"]) is not str or request["capability"] not in CAPABILITIES:
        raise ValueError("Unsupported capability")
    if type(request["deadline_ms"]) is not int or not 1 <= request["deadline_ms"] <= 120000:
        raise ValueError("Invalid deadline")
    _fields(request["payload"], ("requirement",))
    requirement = request["payload"]["requirement"]
    if type(requirement) is not str or not 1 <= len(requirement.strip()) <= 512:
        raise ValueError("Invalid requirement")
    return request
def describe():
    return {"schema_version":1,"capabilities":sorted(CAPABILITIES),"deadline_ms_max":120000}
def request_for(requirement, capability):
    return validate({"schema_version":1,"operation_id":uuid.uuid4().hex,"capability":capability,"deadline_ms":120000,"payload":{"requirement":requirement}})
def receipt(operation_id,status,error_code=None,artifact_sha256=None,evidence_ids=()):
    _id(operation_id)
    if status not in STATES:
        raise ValueError("Invalid state")
    if type(evidence_ids) not in (tuple,list) or any(type(x) is not str or not x for x in evidence_ids):
        raise ValueError("Invalid evidence references")
    if artifact_sha256 is not None and (type(artifact_sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}",artifact_sha256)):
        raise ValueError("Invalid fingerprint")
    if status=="COMPLETED" and (not artifact_sha256 or not evidence_ids or error_code is not None):
        raise ValueError("Completion requires verification evidence")
    if status in ("FAILED","DENIED","BLOCKED") and (type(error_code) is not str or not error_code):
        raise ValueError("Missing error code")
    return {"operation_id":operation_id,"status":status,"error_code":error_code,"artifact_sha256":artifact_sha256,"evidence_ids":list(evidence_ids)}


def trusted_context(actor, authorization, workspace, capability):
    """Check server-supplied context against an already authenticated scope."""
    if actor != "local-owner" or type(workspace) is not str:
        raise ValueError("Invalid trusted context")
    _fields(authorization, ("scope","mac"))
    scope = authorization["scope"]
    if type(scope) is not dict or scope.get("actor") != actor or scope.get("workspace") != workspace or scope.get("capability") != capability:
        raise ValueError("Trusted context mismatch")
    return {"actor":actor,"authorization":authorization,"workspace":workspace}
