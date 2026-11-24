"""Advisory-only S-Brain integration. No generated text is executed."""
from continuum import s_brain_bridge

def propose(requirement, *, timeout=120.0):
    if not isinstance(requirement,str) or not requirement.strip():
        raise ValueError("requirement must contain text")
    result=s_brain_bridge.generate(requirement,timeout=timeout)
    return {"provider":"s-brain:experimental","proposal":result["response"],
            "experimental":True,"authorized":False,"verified":False}

def brain_status():
    return s_brain_bridge.read_health()

def ml_status(store, mission):
    """Resolve only the existing S_ML promotion registry; never train or promote."""
    from s_ml.promotion import active_model
    active=active_model(store,mission)
    return {"mission":mission,"approved_model_available":active is not None,
            "automatic_execution":False}
