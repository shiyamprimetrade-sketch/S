"""Versioned checkpoint management and reproducibility ledger.

Weights stored with safetensors, optimizer state optionally stored with torch.save.
Only load optimizer state from trusted local checkpoints. Inference never needs
the optimizer pickle. Atomic replacements avoid partially written weight files.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = 1
WEIGHT_FILE = "model.safetensors"
META_FILE = "manifest.json"
OPTIM_FILE = "optimizer.pt"


@dataclass(frozen=True)
class CheckpointManifest:
    schema: int
    model_config: dict
    step: int
    train_tokens: int
    train_loss: float
    validation_loss: float | None
    seed: int
    dataset_fingerprint: str
    weights_sha256: str
    created_utc: str
    runtime: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, value: dict) -> "CheckpointManifest":
        if value.get("schema") != SCHEMA:
            raise ValueError("Unsupported checkpoint schema")
        valid = {key: value[key] for key in cls.__dataclass_fields__ if key in value}
        instance = cls(**valid)
        if instance.step < 0 or instance.train_tokens < 0:
            raise ValueError("Invalid training counters")
        if len(instance.weights_sha256) != 64:
            raise ValueError("Invalid weight digest")
        return instance


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_files(paths: list[Path]) -> str:
    """Stable fingerprint of ordered dataset contents, not their absolute paths."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def runtime_info() -> dict:
    import torch
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def utc_string() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(value)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmpname, path)
    finally:
        if os.path.exists(tmpname):
            os.unlink(tmpname)


def save_checkpoint(
    directory: Path,
    model: Any,
    config: Any,
    *,
    step: int,
    train_tokens: int,
    train_loss: float,
    validation_loss: float | None,
    seed: int,
    dataset_fingerprint: str,
    optimizer: Any = None,
    extras: dict | None = None,
) -> CheckpointManifest:
    """Save a self-describing checkpoint in its own output directory.

    Caller should use a different directory per checkpoint when retaining history.
    """
    import torch
    from safetensors.torch import save_model

    if step < 0 or train_tokens < 0 or not dataset_fingerprint:
        raise ValueError("Missing checkpoint counters or dataset fingerprint")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / WEIGHT_FILE
    temporary = directory / (WEIGHT_FILE + ".part")
    try:
        save_model(model, str(temporary))
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    weights_digest = sha256_file(target)
    cfg = asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config)
    result = CheckpointManifest(
        schema=SCHEMA,
        model_config=cfg,
        step=step,
        train_tokens=train_tokens,
        train_loss=float(train_loss),
        validation_loss=None if validation_loss is None else float(validation_loss),
        seed=seed,
        dataset_fingerprint=dataset_fingerprint,
        weights_sha256=weights_digest,
        created_utc=utc_string(),
        runtime=runtime_info(),
        extras=extras or {},
    )
    # Manifest is published last, so incomplete writes are not considered valid.
    if optimizer is not None:
        optimizer_path = directory / OPTIM_FILE
        temporary_optimizer = directory / (OPTIM_FILE + ".part")
        try:
            torch.save(optimizer.state_dict(), temporary_optimizer)
            os.replace(temporary_optimizer, optimizer_path)
        finally:
            if temporary_optimizer.exists():
                temporary_optimizer.unlink()
    _atomic_text(directory / META_FILE, json.dumps(asdict(result), indent=2, sort_keys=True))
    return result


def load_manifest(directory: Path, verify_digest: bool = True) -> CheckpointManifest:
    manifest = CheckpointManifest.parse(
        json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    )
    weights = directory / WEIGHT_FILE
    if not weights.exists():
        raise FileNotFoundError("Checkpoint weights missing")
    if verify_digest and sha256_file(weights) != manifest.weights_sha256:
        raise ValueError("Checkpoint digest mismatch")
    return manifest


def load_weights(directory: Path, device: str = "cpu"):
    """Load safe weights from disk into a newly constructed model."""
    from safetensors.torch import load_model
    from .model import Config, SBrain
    manifest = load_manifest(directory)
    config = Config(**manifest.model_config)
    model = SBrain(config).to(device)
    load_model(model, str(directory / WEIGHT_FILE), strict=True)
    model.eval()
    return model, manifest


def resume_training(directory: Path, model, optimizer,
                    *, trusted_local_optimizer: bool = False) -> CheckpointManifest:
    """Only opt in to loading torch optimizer pickles from files you trust."""
    import torch
    from safetensors.torch import load_model
    manifest = load_manifest(directory)
    if manifest.model_config != asdict(model.cfg):
        raise ValueError("Checkpoint architecture differs from model")
    load_model(model, str(directory / WEIGHT_FILE), strict=True)
    path = directory / OPTIM_FILE
    if path.exists():
        if not trusted_local_optimizer:
            raise PermissionError("Explicit trusted_local_optimizer required")
        state = torch.load(path, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state)
    return manifest


def list_checkpoints(root: Path) -> list[dict]:
    """List valid and invalid checkpoint directories without loading model weights."""
    result = []
    if not root.exists():
        return result
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder / META_FILE).exists():
            continue
        try:
            meta = load_manifest(folder, verify_digest=True)
            result.append({
                "path": str(folder),
                "step": meta.step,
                "validation_loss": meta.validation_loss,
                "valid": True,
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.append({
                "path": str(folder),
                "valid": False,
                "reason": type(exc).__name__,
            })
    return result


def best_checkpoint(root: Path) -> Path | None:
    valid = [
        entry for entry in list_checkpoints(root)
        if entry["valid"] and entry["validation_loss"] is not None
    ]
    if not valid:
        return None
    return Path(min(valid, key=lambda row: row["validation_loss"])["path"])


def remove_old_checkpoints(root: Path, keep: int = 5,
                           dry_run: bool = True) -> list[Path]:
    """Optional retention helper; defaults to preview and never removes best."""
    if keep < 1:
        raise ValueError("keep must be positive")
    candidates = [item for item in list_checkpoints(root) if item["valid"]]
    candidates.sort(key=lambda item: item["step"], reverse=True)
    best = best_checkpoint(root)
    protected = {Path(item["path"]) for item in candidates[:keep]}
    if best is not None:
        protected.add(best)
    removed = []
    for item in candidates:
        path = Path(item["path"])
        if path in protected:
            continue
        removed.append(path)
        if not dry_run:
            shutil.rmtree(path)
    return removed


def checkpoint_summary(directory: Path) -> dict:
    meta = load_manifest(directory)
    result = asdict(meta)
    result["weight_bytes"] = (directory / WEIGHT_FILE).stat().st_size
    result["has_optimizer"] = (directory / OPTIM_FILE).exists()
    return result


def compare_checkpoints(left: Path, right: Path) -> dict:
    a, b = load_manifest(left), load_manifest(right)
    if a.dataset_fingerprint != b.dataset_fingerprint:
        return {"comparable": False, "reason": "different dataset fingerprints"}
    if a.model_config != b.model_config:
        return {"comparable": False, "reason": "different architectures"}
    delta = None
    if a.validation_loss is not None and b.validation_loss is not None:
        delta = b.validation_loss - a.validation_loss
    return {
        "comparable": True,
        "step_delta": b.step - a.step,
        "validation_loss_delta": delta,
        "weights_changed": a.weights_sha256 != b.weights_sha256,
    }
