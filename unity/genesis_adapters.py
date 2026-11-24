"""Explicit advisory and observation interfaces for Genesis."""
from .intelligence import propose
from .interaction import observe
from .requirements import specify

def advisory(requirement,*,enabled=False,provider=propose):
    spec=specify(requirement)
    if enabled is not True:
        return {"status":"disabled","specification":spec,"authorized":False}
    result=provider(requirement)
    proposal=result.get("proposal")
    if not isinstance(proposal,str) or len(proposal)>16000:
        raise ValueError("invalid advisory response")
    return {"status":"advisory","specification":spec,"proposal":proposal,
            "authorized":False,"verified":False}

def inspect_image(image,*,enabled=False,analyzer):
    if enabled is not True:
        return {"status":"disabled","physical_control":False}
    return {"status":"observed","observation":observe(image,analyzer),
            "physical_control":False,"verified":False}

def hand_status():
    return {"status":"unconnected","physical_control":False,"automatic_execution":False}
