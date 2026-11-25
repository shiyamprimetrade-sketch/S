import threading, time, urllib.request
from project_s.local_ui import serve
def start(root):
    thread=threading.Thread(target=serve,args=(root,8768),daemon=True)
    thread.start()
    for _ in range(100):
        if not thread.is_alive(): raise RuntimeError("Engine stopped")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8768/health",timeout=0.2) as r:
                if r.status==200: return "http://127.0.0.1:8768/"
        except OSError: time.sleep(0.1)
    raise RuntimeError("Engine readiness timeout")
