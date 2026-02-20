#!/usr/bin/env python3
"""Minimal pipeline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union


def check_dependencies(dependencies: Dict[str, Union[str, Path]], stop_on_missing: bool = True) -> bool:
    missing = [(k, Path(v)) for k, v in dependencies.items() if not Path(v).exists()]
    if missing and stop_on_missing:
        msg = "; ".join(f"{k}={p}" for k, p in missing)
        raise FileNotFoundError(f"Missing dependencies: {msg}")
    return not missing


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p
