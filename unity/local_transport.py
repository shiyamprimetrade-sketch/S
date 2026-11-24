"""Local Unity v3 transport; no sockets or implicit authorization."""
import hashlib
import hmac
import json
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")

def execute_local(task, signature, key, root, db, *, allowed=(), executor=None):
    """Caller supplies authorization and trusted local executor.
    This is not a sandbox or independent semantic verifier.
    """
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("invalid key")
    raw = canonical(task)
    if len(raw) > 65536:
        raise ValueError("request too large")
    snapshot = json.loads(raw)
    if (not isinstance(snapshot, dict) or
        set(snapshot) != {"version", "id", "operation", "args"} or
        type(snapshot["version"]) is not int or snapshot["version"] != 3 or
        not isinstance(snapshot["id"], str) or
        not isinstance(snapshot["operation"], str) or
        not isinstance(snapshot["args"], dict) or
        snapshot["operation"] not in allowed):
        raise ValueError("invalid or denied request")
    expected = hmac.new(key, b"alpha-fusion-v3\0"+raw, hashlib.sha256).hexdigest()
    if not isinstance(signature, str) or not signature.isascii() or not hmac.compare_digest(expected, signature):
        raise ValueError("authentication failed")
    if executor is None:
        from s_alpha.fusion_bridge import execute_v3
        executor = execute_v3
    evidence = executor(snapshot, signature, key, Path(root), Path(db), allowed=tuple(allowed))
    record = evidence["record"]
    tag = evidence["tag"]
    expected_tag = hmac.new(key, b"alpha-fusion-result-v3\0"+canonical(record), hashlib.sha256).hexdigest()
    if (not isinstance(tag, str) or not tag.isascii() or
        not hmac.compare_digest(expected_tag, tag) or
        record.get("id") != snapshot["id"] or
        record.get("request_digest") != hashlib.sha256(raw).hexdigest()):
        raise ValueError("invalid or mismatched evidence")
    return evidence
