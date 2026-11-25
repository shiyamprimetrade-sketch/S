from contextlib import contextmanager
from pathlib import Path
import os

@contextmanager
def workspace_lock(root):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    path=root/'server.lock'
    if path.is_symlink():raise ValueError('Unsafe lock path')
    with open(path,'a+b') as f:
        f.seek(0,os.SEEK_END)
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
