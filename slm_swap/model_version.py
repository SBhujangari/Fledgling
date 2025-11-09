"""Helper for generating model version identifiers tied to training context."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict


def generate_model_id(
    hyperparams: Dict[str, Any],
    dataset_info: Dict[str, Any],
    timestamp: str | None = None,
) -> str:
    """Return a short SHA256-derived identifier for a training run."""
    ts = timestamp or datetime.utcnow().isoformat()
    payload = {
        "timestamp": ts,
        "hyperparams": hyperparams,
        "dataset": dataset_info,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
