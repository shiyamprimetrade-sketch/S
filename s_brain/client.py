"""Local S-Brain streaming client and diagnostic command-line interface.

Uses the Python standard library. It never sends data to a cloud host and
rejects non-loopback URLs. Supported transports: SSE, NDJSON and single JSON.
The client is intentionally separate from S Continuum's current chat UI.
"""
from __future__ import annotations

import argparse
import codecs
import json
import sys
import time
from dataclasses import dataclass
from typing import Iterator
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit
from urllib.request import Request,urlopen


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = "http://127.0.0.1:8766"
    timeout: float = 120.0
    max_event_bytes: int = 128*1024

    def validate(self):
        parsed = urlsplit(self.base_url)
        if parsed.scheme!="http" or parsed.hostname not in ("127.0.0.1","localhost"):
            raise ValueError("Only explicit loopback HTTP URLs allowed")
        if parsed.username or parsed.password or parsed.fragment or parsed.query:
            raise ValueError("URL may not contain credentials or extra components")
        if not 0 < self.timeout <= 3600:
            raise ValueError("Timeout out of range")
        if not 1024 <= self.max_event_bytes <= 4*1024*1024:
            raise ValueError("Event limit out of range")


class StreamProtocolError(RuntimeError):
    pass


class ServiceUnavailable(RuntimeError):
    pass


def parse_sse(lines: Iterator[bytes],max_event_bytes: int = 128*1024) -> Iterator[dict]:
    """Parse SSE data/event frames, preserving text inside JSON data payloads."""
    event_type = "message"
    data = []
    size = 0
    for raw in lines:
        if not isinstance(raw,bytes):
            raise TypeError("SSE parser requires bytes")
        size += len(raw)
        if size > max_event_bytes:
            raise StreamProtocolError("SSE frame exceeded maximum size")
        line = raw.decode("utf-8",errors="strict").rstrip("\r\n")
        if line == "":
            if data:
                raw_data = "\n".join(data)
                try:
                    event = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise StreamProtocolError("Malformed SSE JSON") from exc
                if not isinstance(event,dict):
                    raise StreamProtocolError("SSE payload must be object")
                event.setdefault("type",event_type)
                yield event
            event_type = "message"
            data = []
            size = 0
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        raise StreamProtocolError("SSE stream truncated before final blank line")


def parse_ndjson(lines: Iterator[bytes],
                 max_event_bytes: int = 128*1024) -> Iterator[dict]:
    for raw in lines:
        if len(raw) > max_event_bytes:
            raise StreamProtocolError("NDJSON event too large")
        if not raw.strip():
            continue
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeError,json.JSONDecodeError) as exc:
            raise StreamProtocolError("Invalid NDJSON event") from exc
        if not isinstance(event,dict):
            raise StreamProtocolError("NDJSON event must be an object")
        yield event


class SBrainClient:
    def __init__(self,config: ClientConfig | None = None):
        self.config = config or ClientConfig()
        self.config.validate()

    def endpoint(self,path: str):
        if not path.startswith("/") or "://" in path:
            raise ValueError("Invalid endpoint path")
        return self.config.base_url.rstrip("/")+path

    def health(self) -> dict:
        req = Request(self.endpoint("/health"),headers={"Accept":"application/json"})
        try:
            with urlopen(req,timeout=self.config.timeout) as response:
                return json.load(response)
        except (URLError,TimeoutError,OSError) as exc:
            raise ServiceUnavailable("Local S-Brain service unavailable") from exc

    def generate(self,prompt: str,**kwargs) -> dict:
        body = dict(kwargs,prompt=prompt)
        req = self._request("/generate",body)
        try:
            with urlopen(req,timeout=self.config.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise ServiceUnavailable(f"S-Brain returned HTTP {exc.code}") from exc
        except (URLError,TimeoutError,OSError) as exc:
            raise ServiceUnavailable("Local S-Brain service unavailable") from exc
        if not isinstance(result,dict) or "response" not in result:
            raise StreamProtocolError("Invalid generation response")
        return result

    def stream(self,prompt: str,transport: str = "sse",**kwargs) -> Iterator[dict]:
        if transport not in ("sse","ndjson"):
            raise ValueError("Transport must be sse or ndjson")
        path = "/generate/stream" if transport=="sse" else "/generate/ndjson"
        req = self._request(path,dict(kwargs,prompt=prompt))
        try:
            with urlopen(req,timeout=self.config.timeout) as response:
                content_type = response.headers.get("Content-Type","")
                expected = "text/event-stream" if transport=="sse" else "application/x-ndjson"
                if not content_type.startswith(expected):
                    raise StreamProtocolError("Unexpected streaming content type")
                parse = parse_sse if transport=="sse" else parse_ndjson
                finished = False
                sequence = -1
                for event in parse(response,self.config.max_event_bytes):
                    if event.get("sequence",sequence+1) <= sequence:
                        raise StreamProtocolError("Non-monotonic stream sequence")
                    sequence = event.get("sequence",sequence+1)
                    if event.get("type") == "complete":
                        finished = True
                    yield event
                if not finished:
                    raise StreamProtocolError("Stream ended before completion")
        except HTTPError as exc:
            raise ServiceUnavailable(f"S-Brain returned HTTP {exc.code}") from exc
        except (URLError,TimeoutError,OSError) as exc:
            raise ServiceUnavailable("Local streaming service unavailable") from exc

    def _request(self,path: str,body: dict) -> Request:
        data = json.dumps(body,ensure_ascii=False,allow_nan=False).encode("utf-8")
        return Request(
            self.endpoint(path),data=data,method="POST",
            headers={"Content-Type":"application/json","Accept":"application/json"}
        )

    def stream_text(self,prompt: str,**kwargs) -> Iterator[str]:
        for event in self.stream(prompt,**kwargs):
            if event.get("type")=="delta":
                yield event.get("text","")


def collect_text(client: SBrainClient,prompt: str,**kwargs) -> str:
    return "".join(client.stream_text(prompt,**kwargs))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="S-Brain local streaming client")
    p.add_argument("--url",default="http://127.0.0.1:8766")
    p.add_argument("--timeout",type=float,default=120)
    sub = p.add_subparsers(dest="command",required=True)
    sub.add_parser("health")
    gen = sub.add_parser("generate")
    gen.add_argument("prompt")
    gen.add_argument("--tokens",type=int,default=64)
    gen.add_argument("--temperature",type=float,default=0.8)
    gen.add_argument("--seed",type=int)
    stream = sub.add_parser("stream")
    stream.add_argument("prompt")
    stream.add_argument("--tokens",type=int,default=64)
    stream.add_argument("--transport",choices=("sse","ndjson"),default="sse")
    stream.add_argument("--temperature",type=float,default=0.8)
    stream.add_argument("--seed",type=int)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    client = SBrainClient(ClientConfig(args.url,args.timeout))
    try:
        if args.command=="health":
            print(json.dumps(client.health(),indent=2))
            return 0
        params = {"max_new_tokens":args.tokens,"temperature":args.temperature}
        if args.seed is not None:
            params["seed"] = args.seed
        if args.command=="generate":
            print(json.dumps(client.generate(args.prompt,**params),indent=2))
            return 0
        for event in client.stream(args.prompt,transport=args.transport,**params):
            if event.get("type")=="delta":
                sys.stdout.write(event.get("text",""))
                sys.stdout.flush()
            if event.get("type")=="complete":
                print()
                print(json.dumps({"usage":event.get("usage"),"finish_reason":event.get("finish_reason")},indent=2))
        return 0
    except (ValueError,StreamProtocolError,ServiceUnavailable) as exc:
        print(f"S-Brain client: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
