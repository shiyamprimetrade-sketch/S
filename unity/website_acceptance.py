"""Independent acceptance for the narrow static website contract."""
from html.parser import HTMLParser
from pathlib import Path
from hashlib import sha256

class ContractParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack=[]
        self.values={"title":"","h1":"","p":""}
        self.tags=[]
        self.valid=True
    def handle_starttag(self,tag,attrs):
        self.tags.append(tag)
        allowed={"html":{"lang":"en"},"meta":None}
        if tag not in ("html","head","meta","title","body","main","h1","p"):
            self.valid=False
        if tag=="meta":
            if self.stack!=["html","head"] or dict(attrs) not in ({"charset":"utf-8"},{"name":"viewport","content":"width=device-width, initial-scale=1"}):
                self.valid=False
            return
        if dict(attrs)!=allowed.get(tag,{}):
            self.valid=False
        self.stack.append(tag)
    def handle_endtag(self,tag):
        if not self.stack or self.stack.pop()!=tag:
            self.valid=False
    def handle_data(self,data):
        if self.stack and self.stack[-1] in self.values:
            self.values[self.stack[-1]]+=data
        elif data.strip():
            self.valid=False
    def handle_comment(self,data):
        self.valid=False
    def handle_startendtag(self,tag,attrs):
        self.valid=False

def verify_website(root,name,heading,description):
    path=Path(root)/"index.html"
    if path.is_symlink() or not path.is_file() or path.stat().st_size>65536:
        return {"status":"rejected","reason":"invalid artifact"}
    data=path.read_bytes()
    parser=ContractParser()
    try:
        parser.feed(data.decode("utf-8"))
        parser.close()
    except (ValueError,UnicodeError):
        return {"status":"rejected","reason":"invalid HTML"}
    expected=["html","head","meta","meta","title","body","main","h1","p"]
    ok=(parser.valid and not parser.stack and parser.tags==expected and
        parser.values=={"title":name,"h1":heading,"p":description})
    return {"status":"verified" if ok else "rejected","sha256":sha256(data).hexdigest(),
            "contract":"static-website-v1"}
