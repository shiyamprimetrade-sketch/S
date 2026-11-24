"""Verify extracted preview files before launch; manifest is not a signature."""
import hashlib
import json
from pathlib import Path, PurePosixPath

def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/"SHA256.json").read_text(encoding="utf-8"))
    if not isinstance(manifest,dict) or not manifest:
        raise ValueError("empty manifest")
    for name,digest in manifest.items():
        rel=PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
            raise ValueError("unsafe manifest path")
        p=root.joinpath(*rel.parts)
        if not p.resolve().is_relative_to(root) or not p.is_file() or p.is_symlink():
            raise ValueError("missing or unsafe file: "+name)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError("digest mismatch: "+name)
    return len(manifest)

if __name__=="__main__":
    print("Verified files:",verify(Path.cwd()))
