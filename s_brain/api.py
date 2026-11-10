"""Loopback-only standalone S-Brain API, port 8766 (S uses 8765).

Start: python -m s_brain.api --checkpoint s_brain/checkpoints
This API does not modify or replace the existing S server.
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from .model import Config, SBrain

def build_handler(model, device):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, value):
            raw = json.dumps(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)
        def allowed(self):
            return self.headers.get("Host") in ("127.0.0.1:8766", "localhost:8766") and (
                not self.headers.get("Origin") or self.headers["Origin"] in ("http://127.0.0.1:8766", "http://localhost:8766"))
        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {"error":"Invalid host/origin"})
            if self.path == "/health":
                return self.reply(200, {"ready":True,"model":"s-brain:experimental","independent":True})
            self.reply(404, {"error":"Not found"})
        def do_POST(self):
            if not self.allowed():
                return self.reply(403, {"error":"Invalid host/origin"})
            if self.path != "/generate":
                return self.reply(404, {"error":"Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 20000:
                    raise ValueError("Invalid request length")
                body = json.loads(self.rfile.read(length))
                text = body["prompt"]
                if not isinstance(text,str) or not text or len(text)>4000:
                    raise ValueError("Prompt must be 1-4000 characters")
                steps = int(body.get("max_new_tokens", 64))
                if not 1 <= steps <= 512:
                    raise ValueError("max_new_tokens must be 1-512")
                ids = torch.tensor([list(text.encode("utf-8"))[-model.cfg.context:]], device=device)
                result = model.generate(ids, steps=steps)
                answer = bytes(result[0,len(ids[0]):].tolist()).decode("utf-8", errors="replace")
                self.reply(200, {"response":answer,"model":"s-brain:experimental"})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.reply(400, {"error":str(exc)})
        def log_message(self, *args):
            pass
    return Handler

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,default=Path("s_brain/checkpoints"))
    args = p.parse_args()
    cfg = Config(**json.loads((args.checkpoint/"config.json").read_text()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBrain(cfg).to(device)
    # Only load checkpoints produced locally or from a trusted source.
    state = torch.load(args.checkpoint/"weights.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    server = ThreadingHTTPServer(("127.0.0.1",8766),build_handler(model,device))
    print("S-Brain standalone API: http://127.0.0.1:8766", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
