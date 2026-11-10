"""Bounded fair microbatch scheduler for S-Brain inference.

This component admits requests into a queue, groups compatible pending
requests, executes the real grouped batch decoder, and delivers individual
results through futures. It is a microbatch *admission* scheduler, not a
continuous in-flight token-level batching engine. Already-running batches
finish before the next group begins.

Runtime is explicit: start(), submit(), await_result(), stop(). No daemon
starts on module import and no remote execution tools are exposed.
"""
from __future__ import annotations

import collections
import dataclasses
import threading
import time
from concurrent.futures import Future
from typing import Callable

from .batching import BatchRequest, BatchResult, batch_generate


class QueueFull(RuntimeError):
    pass


class SchedulerClosed(RuntimeError):
    pass


class RequestExpired(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class SchedulerPolicy:
    max_queue: int = 128
    max_batch_size: int = 8
    batch_wait_ms: float = 10.0
    max_wait_seconds: float = 60.0
    max_pending_per_owner: int = 32
    max_prompt_bytes: int = 8192
    max_new_tokens: int = 256

    def validate(self):
        if not 1 <= self.max_queue <= 100000:
            raise ValueError("Invalid queue size")
        if not 1 <= self.max_batch_size <= 1024:
            raise ValueError("Invalid batch size")
        if not 0 <= self.batch_wait_ms <= 5000:
            raise ValueError("Invalid batch wait")
        if not 0.01 <= self.max_wait_seconds <= 3600:
            raise ValueError("Invalid maximum wait")
        if not 1 <= self.max_pending_per_owner <= self.max_queue:
            raise ValueError("Invalid owner quota")
        if self.max_prompt_bytes < 1 or self.max_new_tokens < 1:
            raise ValueError("Invalid request limits")


@dataclasses.dataclass
class Pending:
    request: BatchRequest
    owner: str
    future: Future
    enqueued_at: float
    deadline: float
    state: str = "queued"

    @property
    def length(self):
        return len(self.request.prompt.encode("utf-8"))

    def expired(self, now: float):
        return now > self.deadline


@dataclasses.dataclass
class SchedulerStats:
    submitted: int = 0
    rejected: int = 0
    cancelled: int = 0
    expired: int = 0
    completed: int = 0
    failed: int = 0
    batches: int = 0
    batched_requests: int = 0
    total_queue_seconds: float = 0.0

    def report(self, depth: int, running: bool) -> dict:
        return {
            **dataclasses.asdict(self),
            "queue_depth": depth,
            "running": running,
            "average_batch_size": self.batched_requests / max(1, self.batches),
            "mean_queue_seconds": self.total_queue_seconds / max(1, self.batched_requests),
        }


class FairMicrobatchScheduler:
    """Round-robin service across owners with equal-length grouping.

    A short batching window allows compatible requests to arrive together.
    Fairness comes from owner round-robin, not arbitrary priority labels.
    The worker is single-threaded by design: model weights aren't mutated
    while inference is being served by this scheduler.
    """

    def __init__(
        self,
        model,
        *,
        policy: SchedulerPolicy | None = None,
        device: str | None = None,
        executor: Callable | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.model = model
        self.policy = policy or SchedulerPolicy()
        self.policy.validate()
        self.device = device
        self.executor = executor or self._execute
        self.clock = clock
        self._condition = threading.Condition()
        self._owners: dict[str, collections.deque[Pending]] = {}
        self._round_robin: collections.deque[str] = collections.deque()
        self._entries: dict[str, Pending] = {}
        self._running = False
        self._closing = False
        self._worker: threading.Thread | None = None
        self._stats = SchedulerStats()

    def _execute(self, requests: list[BatchRequest]) -> list[BatchResult]:
        return batch_generate(
            self.model, requests, device=self.device,
            max_batch_size=self.policy.max_batch_size
        )

    def start(self):
        with self._condition:
            if self._closing:
                raise SchedulerClosed("Scheduler is closed")
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(
                target=self._loop, name="s-brain-microbatch", daemon=True
            )
            self._worker.start()

    def submit(
        self,
        request: BatchRequest,
        *,
        owner: str = "default",
        timeout: float | None = None,
    ) -> Future:
        request.validate(self.model.cfg.context)
        if not owner or len(owner) > 128:
            raise ValueError("Invalid owner")
        if len(request.prompt.encode("utf-8")) > self.policy.max_prompt_bytes:
            raise ValueError("Prompt exceeds scheduler allowance")
        if request.max_new_tokens > self.policy.max_new_tokens:
            raise ValueError("Output length exceeds scheduler allowance")
        window = self.policy.max_wait_seconds if timeout is None else timeout
        if window <= 0:
            raise ValueError("Timeout must be positive")
        future = Future()
        now = self.clock()
        pending = Pending(request, owner, future, now, now + window)
        with self._condition:
            if not self._running or self._closing:
                raise SchedulerClosed("Scheduler must be running")
            depth = sum(len(queue) for queue in self._owners.values())
            owner_queue = self._owners.get(owner, ())
            if depth >= self.policy.max_queue or len(owner_queue) >= self.policy.max_pending_per_owner:
                self._stats.rejected += 1
                raise QueueFull("Scheduling quota reached")
            if request.identifier in self._entries:
                raise ValueError("Duplicate pending request identifier")
            if owner not in self._owners:
                self._owners[owner] = collections.deque()
                self._round_robin.append(owner)
            self._owners[owner].append(pending)
            self._entries[request.identifier] = pending
            self._stats.submitted += 1
            self._condition.notify_all()
        return future

    def cancel(self, identifier: str) -> bool:
        with self._condition:
            pending = self._entries.get(identifier)
            if pending is None or pending.state != "queued":
                return False
            pending.state = "cancelled"
            pending.future.cancel()
            self._entries.pop(identifier, None)
            self._stats.cancelled += 1
            self._condition.notify_all()
            return True

    def _purge_front(self, owner: str):
        queue = self._owners.get(owner)
        if not queue:
            return
        now = self.clock()
        while queue and (
            queue[0].state == "cancelled" or queue[0].expired(now)
        ):
            entry = queue.popleft()
            self._entries.pop(entry.request.identifier, None)
            if entry.state != "cancelled":
                entry.state = "expired"
                self._stats.expired += 1
                if not entry.future.done():
                    entry.future.set_exception(RequestExpired("Request deadline exceeded"))
        if not queue:
            self._owners.pop(owner, None)
            try:
                self._round_robin.remove(owner)
            except ValueError:
                pass

    def _select_batch(self) -> list[Pending]:
        """Rotate owners and prefer matching effective lengths for one batch."""
        chosen = []
        target_length = None
        owners = list(self._round_robin)
        for owner in owners:
            self._purge_front(owner)
        owners = list(self._round_robin)
        if not owners:
            return []
        for owner in owners:
            queue = self._owners.get(owner)
            if not queue:
                continue
            entry = queue[0]
            length = min(entry.length, self.model.cfg.context)
            if target_length is None:
                target_length = length
            if length != target_length:
                continue
            entry = queue.popleft()
            entry.state = "running"
            self._entries.pop(entry.request.identifier, None)
            chosen.append(entry)
            self._stats.total_queue_seconds += max(0, self.clock() - entry.enqueued_at)
            if owner in self._round_robin:
                self._round_robin.remove(owner)
            if queue:
                self._round_robin.append(owner)
            else:
                self._owners.pop(owner, None)
            if len(chosen) >= self.policy.max_batch_size:
                break
        # Fill the remaining group from owners who have same-length work.
        progress = True
        while progress and len(chosen) < self.policy.max_batch_size:
            progress = False
            for owner in list(self._round_robin):
                self._purge_front(owner)
                queue = self._owners.get(owner)
                if not queue:
                    continue
                entry = queue[0]
                if min(entry.length, self.model.cfg.context) != target_length:
                    continue
                entry = queue.popleft()
                entry.state = "running"
                self._entries.pop(entry.request.identifier, None)
                chosen.append(entry)
                self._stats.total_queue_seconds += max(0, self.clock() - entry.enqueued_at)
                self._round_robin.remove(owner)
                if queue:
                    self._round_robin.append(owner)
                else:
                    self._owners.pop(owner, None)
                progress = True
                if len(chosen) >= self.policy.max_batch_size:
                    break
        return chosen

    def _loop(self):
        while True:
            with self._condition:
                while not self._owners and not self._closing:
                    self._condition.wait(timeout=0.1)
                if self._closing and not self._owners:
                    self._running = False
                    return
                if self.policy.batch_wait_ms:
                    self._condition.wait(timeout=self.policy.batch_wait_ms / 1000)
                batch = self._select_batch()
            if not batch:
                continue
            self._stats.batches += 1
            self._stats.batched_requests += len(batch)
            try:
                responses = self.executor([item.request for item in batch])
                if len(responses) != len(batch):
                    raise RuntimeError("Executor returned wrong batch size")
                for item, response in zip(batch, responses):
                    if response.identifier != item.request.identifier:
                        raise RuntimeError("Executor violated request order")
                for item, response in zip(batch, responses):
                    item.state = "complete"
                    if not item.future.done():
                        item.future.set_result(response)
                    self._stats.completed += 1
            except Exception as exc:
                for item in batch:
                    item.state = "failed"
                    if not item.future.done():
                        item.future.set_exception(exc)
                    self._stats.failed += 1

    def stop(self, *, drain: bool = True, timeout: float = 10.0):
        with self._condition:
            if not self._running:
                return
            self._closing = True
            if not drain:
                for owner, queue in list(self._owners.items()):
                    for item in queue:
                        if not item.future.done():
                            item.future.cancel()
                            self._stats.cancelled += 1
                self._owners.clear()
                self._entries.clear()
                self._round_robin.clear()
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
            if worker.is_alive():
                raise TimeoutError("Scheduler worker still processing inference")

    def stats(self) -> dict:
        with self._condition:
            depth = sum(len(queue) for queue in self._owners.values())
            return self._stats.report(depth, self._running)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop(drain=True)


def wait_result(future: Future, timeout: float = 60.0) -> BatchResult:
    result = future.result(timeout=timeout)
    if not isinstance(result, BatchResult):
        raise TypeError("Unexpected batch result")
    return result
