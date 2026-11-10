"""Opt-in standalone S-Brain v2 HTTP server with streaming SSE and NDJSON.

The current S application continues to use port 8765. This service defaults
to port 8766, only binds loopback, validates origin and host and never executes
generated text. Loading advanced safetensors checkpoints is supported without
breaking the older model formats. It is not a public internet service.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict,dataclass,field
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .streaming import (
    GenerationOptions,collect_events,jsonl_event,sse_event,stream_events,
)
from .sampling import SamplingOptions

LOG = logging.getLogger("s_brain")
MODEL_NAME = "s-brain:experimental"
MAX_BODY = 24000


@dataclass
class ServerLimits:
    host: str = "127.0.0.1"
    port: int = 8766
    max_concurrent: int = 1
    max_prompt_bytes: int = 16000
    max_new_tokens: int = 512
    max_body_bytes: int = MAX_BODY
    max_temperature: float = 3.0
    allow_streaming: bool = True

    def validate(self):
        if self.host not in ("127.0.0.1","localhost"):
            raise ValueError("S-Brain server must bind localhost")
        if not 1 <= self.port <= 65535:
            raise ValueError("Port out of range")
        if not 1 <= self.max_concurrent <= 16:
            raise ValueError("max_concurrent out of range")
        if not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens out of range")
        if not 1 <= self.max_body_bytes <= 2_000_000:
            raise ValueError("Invalid request size limit")
        if self.max_prompt_bytes < 1 or self.max_prompt_bytes >= self.max_body_bytes:
            raise ValueError("Invalid prompt size cap")


class BusyError(RuntimeError):
    pass


class AdmissionControl:
    """Refuse excess requests rather than accumulating unbounded decode work."""

    def __init__(self,capacity: int):
        if capacity < 1:
            raise ValueError("Admission capacity must be positive")
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._capacity = capacity
        self._lock = threading.Lock()
        self._active = 0
        self._accepted = 0
        self._rejected = 0
        self._completed = 0

    def acquire(self):
        if not self._semaphore.acquire(blocking=False):
            with self._lock:
                self._rejected += 1
            raise BusyError("Model is busy")
        with self._lock:
            self._active += 1
            self._accepted += 1

    def release(self):
        with self._lock:
            if self._active < 1:
                raise RuntimeError("Admission release without acquisition")
            self._active -= 1
            self._completed += 1
        self._semaphore.release()

    def snapshot(self):
        with self._lock:
            return {
                "capacity":self._capacity,
                "active":self._active,
                "accepted":self._accepted,
                "rejected":self._rejected,
                "completed":self._completed,
            }


@dataclass(frozen=True)
class RequestParameters:
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int | None = None
    stop: tuple[str,...] = ()
    use_cache: bool = True

    @classmethod
    def parse(cls,value: object,limits: ServerLimits):
        if not isinstance(value,dict):
            raise ValueError("JSON object required")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value)-allowed
        if unknown:
            raise ValueError("Unknown request fields: " + ",".join(sorted(unknown)))
        if "prompt" not in value:
            raise ValueError("Missing prompt")
        prompt = value["prompt"]
        if not isinstance(prompt,str) or not prompt:
            raise ValueError("Prompt must be nonempty string")
        if len(prompt.encode("utf-8"))>limits.max_prompt_bytes:
            raise ValueError("Prompt too large")
        fields = dict(value)
        if "stop" in fields:
            if not isinstance(fields["stop"],list) or not all(
                isinstance(x,str) for x in fields["stop"]
            ):
                raise ValueError("stop must be an array of strings")
            fields["stop"] = tuple(fields["stop"])
        if "use_cache" in fields and not isinstance(fields["use_cache"],bool):
            raise ValueError("use_cache must be boolean")
        integer_fields = ("max_new_tokens","top_k")
        for name in integer_fields:
            if name in fields and (type(fields[name]) is not int):
                raise ValueError(f"{name} must be integer")
        if "seed" in fields and fields["seed"] is not None and type(fields["seed"]) is not int:
            raise ValueError("seed must be integer")
        float_fields = (
            "temperature","top_p","min_p","repetition_penalty",
            "frequency_penalty","presence_penalty"
        )
        for name in float_fields:
            if name in fields and (
                type(fields[name]) not in (int,float)
            ):
                raise ValueError(f"{name} must be number")
        req = cls(**fields)
        if req.max_new_tokens>limits.max_new_tokens:
            raise ValueError("max_new_tokens exceeds server cap")
        if req.temperature>limits.max_temperature:
            raise ValueError("temperature exceeds server cap")
        req.sampling().validate()
        return req

    def sampling(self) -> SamplingOptions:
        return SamplingOptions(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,top_k=self.top_k,
            top_p=self.top_p,min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            seed=self.seed,stop=self.stop,
        )

    def generation(self,limits: ServerLimits) -> GenerationOptions:
        return GenerationOptions(
            sampling=self.sampling(),
            use_cache=self.use_cache,
            max_prompt_bytes=limits.max_prompt_bytes,
        )


@dataclass
class Runtime:
    model: object
    limits: ServerLimits
    model_source: str = ""
    admitted: AdmissionControl = field(init=False)
    started: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.limits.validate()
        self.admitted = AdmissionControl(self.limits.max_concurrent)

    def health(self):
        return {
            "ready":True,
            "name":MODEL_NAME,
            "independent":True,
            "checkpoint_source":self.model_source,
            "context":self.model.cfg.context,
            "model_parameters":sum(p.numel() for p in self.model.parameters()),
            "uptime_seconds":max(0,time.monotonic()-self.started),
            "admission":self.admitted.snapshot(),
        }


def load_runtime(directory: Path,limits: ServerLimits):
    """Auto-select versioned safetensors or legacy trusted local weights."""
    import torch
    from .checkpoint import load_weights
    from .model import Config,SBrain
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (directory/"manifest.json").exists():
        model,_ = load_weights(directory,device)
        source = "safetensors-v1"
    elif (directory/"config.json").exists() and (directory/"weights.pt").exists():
        cfg = Config(**json.loads((directory/"config.json").read_text("utf-8")))
        model = SBrain(cfg).to(device)
        # This path is intended only for checkpoints created by the owner.
        state = torch.load(directory/"weights.pt",map_location=device,weights_only=True)
        model.load_state_dict(state,strict=True)
        source = "legacy-weights"
    else:
        raise FileNotFoundError("No recognized checkpoint files in " + str(directory))
    model.eval()
    return Runtime(model,limits,source)


def make_handler(runtime: Runtime):
    limits = runtime.limits
    local_hosts = {
        f"127.0.0.1:{limits.port}",f"localhost:{limits.port}",
    }
    local_origins = {f"http://{name}" for name in local_hosts}

    class API(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "S-Brain/0.2"
        def log_message(self,format,*args):
            # Avoid logging private prompt content.
            LOG.debug("Request from localhost: %s",format % args)

        def common_headers(self):
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Cache-Control","no-store")
            self.send_header("Referrer-Policy","no-referrer")
            self.send_header("X-Frame-Options","DENY")
            self.send_header("Access-Control-Allow-Origin","null")
            self.send_header("Connection","close")
            self.close_connection = True

        def send_json(self,code:int,value:dict):
            raw = json.dumps(value,ensure_ascii=False,allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Content-Length",str(len(raw)))
            self.common_headers()
            self.end_headers()
            self.wfile.write(raw)

        def authorized_local_request(self):
            host = self.headers.get("Host","")
            origin = self.headers.get("Origin")
            if host not in local_hosts:
                return False
            if origin and origin not in local_origins:
                return False
            return True

        def read_json(self):
            content_type = self.headers.get("Content-Type","").split(";")[0].strip().lower()
            if content_type != "application/json":
                raise ValueError("Content-Type must be application/json")
            raw_size = self.headers.get("Content-Length")
            if raw_size is None or not raw_size.isdecimal():
                raise ValueError("Content-Length required")
            size = int(raw_size)
            if size < 2 or size > limits.max_body_bytes:
                raise ValueError("Invalid request size")
            raw = self.rfile.read(size)
            if len(raw)!=size:
                raise ValueError("Incomplete request body")
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeError,json.JSONDecodeError) as exc:
                raise ValueError("Invalid JSON document") from exc

        def do_GET(self):
            if not self.authorized_local_request():
                return self.send_json(403,{"error":"Local host/origin required"})
            if urlsplit(self.path).path=="/health":
                return self.send_json(200,runtime.health())
            if urlsplit(self.path).path=="/v1/models":
                return self.send_json(200,{
                    "models":[{"id":MODEL_NAME,"type":"byte-transformer"}]
                })
            return self.send_json(404,{"error":"Not found"})

        def do_POST(self):
            if not self.authorized_local_request():
                return self.send_json(403,{"error":"Local host/origin required"})
            path = urlsplit(self.path).path
            if path not in ("/generate","/generate/stream","/generate/ndjson"):
                return self.send_json(404,{"error":"Not found"})
            if path != "/generate" and not limits.allow_streaming:
                return self.send_json(403,{"error":"Streaming disabled"})
            try:
                req = RequestParameters.parse(self.read_json(),limits)
            except (ValueError,TypeError) as exc:
                return self.send_json(400,{"error":str(exc)})
            try:
                runtime.admitted.acquire()
            except BusyError:
                return self.send_json(429,{"error":"Model busy","retryable":True})
            try:
                self.execute_generation(path,req)
            except (BrokenPipeError,ConnectionResetError):
                # Client disconnected during streaming; release admission slot.
                return
            except (ValueError,RuntimeError) as exc:
                LOG.warning("Generation refused: %s",type(exc).__name__)
                if path=="/generate":
                    return self.send_json(400,{"error":str(exc)})
            finally:
                runtime.admitted.release()

        def execute_generation(self,path,req):
            options = req.generation(limits)
            source = stream_events(runtime.model,req.prompt,options)
            if path=="/generate":
                result = collect_events(source)
                return self.send_json(200,result)

            sse = path=="/generate/stream"
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream; charset=utf-8" if sse
                else "application/x-ndjson; charset=utf-8"
            )
            self.send_header("X-Accel-Buffering","no")
            self.common_headers()
            self.end_headers()
            self.wfile.flush()
            try:
                for event in source:
                    payload = sse_event(event) if sse else jsonl_event(event)
                    self.wfile.write(payload)
                    self.wfile.flush()
            finally:
                source.close()

    return API


def serve(runtime: Runtime):
    with ThreadingHTTPServer(
        (runtime.limits.host,runtime.limits.port),make_handler(runtime)
    ) as server:
        server.daemon_threads = True
        print(f"S-Brain experimental API: http://127.0.0.1:{runtime.limits.port}",flush=True)
        server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description="S-Brain independent streaming API")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--port",type=int,default=8766)
    parser.add_argument("--max-concurrent",type=int,default=1)
    parser.add_argument("--max-new-tokens",type=int,default=512)
    args = parser.parse_args(argv)
    limits = ServerLimits(
        port=args.port,max_concurrent=args.max_concurrent,
        max_new_tokens=args.max_new_tokens
    )
    serve(load_runtime(args.checkpoint,limits))


if __name__=="__main__":
    main()
