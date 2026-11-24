"""Parse a documented subset of English; never infer missing features."""
import re
from .requirements import specify

def extract(text):
    if not isinstance(text,str) or not text.strip() or len(text)>4000:
        raise ValueError("invalid requirement")
    match=re.fullmatch(r'(?:build|create|make) (?:a )?website (?:called|named) "([^"\n]+)" with heading "([^"\n]+)" and description "([^"\n]+)"',text.strip(),re.I)
    if match:
        name,heading,description=match.groups()
        return {"status":"ready","kind":"website","name":name,"heading":heading,
                "description":description,"features":["heading","description"],"authorized":False}
    return specify(text)
