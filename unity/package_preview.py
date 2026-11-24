"""Package trusted pinned checkouts into a portable Python preview."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

def build(root, output):
    root=Path(root).resolve()
    pins=json.loads((root/"unity/baseline.json").read_text())
    entries={}
    for name in ("alpha","omega"):
        checkout=root/name
        actual=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
        if actual!=pins[name]["commit"]:
            raise ValueError("dependency revision mismatch: "+name)
        if subprocess.check_output(["git","-C",str(checkout),"status","--porcelain"],text=True).strip():
            raise ValueError("dirty dependency: "+name)
        files=subprocess.check_output(["git","-C",str(checkout),"ls-files","-z"]).decode().split("\0")
        for rel in files:
            if not rel or not rel.endswith((".py",".md",".txt",".json")):
                continue
            p=checkout/rel
            if p.is_file() and not p.is_symlink():
                entries[name+"/"+rel]=p.read_bytes()
    for p in (root/"unity").glob("*.py"):
        entries["unity/"+p.name]=p.read_bytes()
    entries["unity/baseline.json"]=(root/"unity/baseline.json").read_bytes()
    entries["README.txt"]=b"Project S Unity portable Python preview. Python 3.11+ required. Run python launch.py status or python launch.py calculate 8+9 --approve. Native APK/EXE not included.\n"
    entries["launch.py"]=b"import sys\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path[:0]=[str(root/'alpha'),str(root/'omega')]\nfrom unity.cli import main\nraise SystemExit(main())\n"
    manifest={name:hashlib.sha256(data).hexdigest() for name,data in sorted(entries.items())}
    entries["SHA256.json"]=json.dumps(manifest,sort_keys=True,indent=2).encode()
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    return hashlib.sha256(Path(output).read_bytes()).hexdigest()

if __name__=="__main__":
    print(build(Path.cwd(),"S-Unity-preview.zip"))
