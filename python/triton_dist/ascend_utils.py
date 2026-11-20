################################################################################
#
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
################################################################################
"""Ascend-only helpers, parallel to ``nv_utils.py`` / ``amd_utils.py``.

Imported from ``triton_dist.utils`` when ``npu-smi`` is on PATH.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import os
import re
import subprocess
import warnings
from pathlib import Path

import torch

# Last-resort 910B rated AI Core freq from ascend_910b.json, in MHz.
_FALLBACK_AICORE_MHZ = 1850

_FREQ_CMDS = (
    ("npu-smi", "info", "-t", "frequencies", "-i", "{id}", "-c", "0"),
    ("npu-smi", "info", "-t", "common", "-i", "{id}", "-c", "0"),
    ("npu-smi", "info", "-t", "freq", "-i", "{id}"),
)


def _npu_smi_index(device_id):
    # torch logical device id and npu-smi id are both 0..N-1 and aligned.
    return device_id if device_id is not None else torch.npu.current_device()


def _npu_smi_info(device_id: int) -> str:
    last_err = None
    for tmpl in _FREQ_CMDS:
        cmd = [p.format(id=device_id) if "{id}" in p else p for p in tmpl]
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError) as e:
            last_err = e
    raise RuntimeError(f"npu-smi clock query failed: {last_err}")


def _parse_aicore_clocks_mhz(text: str) -> tuple[int | None, int | None]:
    """Return (current_mhz, rated_mhz) from ``npu-smi`` text. Missing fields are None."""
    current, rated = None, None

    m = re.search(r"AI\s*Core\s+Current\s+Freq[^:]*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        current = int(m.group(1))
    m = re.search(r"AI\s*Core\s+Rated\s+Freq[^:]*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        rated = int(m.group(1))

    m = re.search(
        r"Aicore\s+Freq\s*\(\s*MHZ\s*\)\s*:\s*(\d+)\s*/\s*(\d+)", text, re.IGNORECASE
    )
    if m:
        if current is None:
            current = int(m.group(1))
        if rated is None:
            rated = int(m.group(2))
    m = re.search(
        r"Aicore\s+Freq\s+Rated\s*\(\s*MHZ\s*\)\s*:\s*(\d+)", text, re.IGNORECASE
    )
    if m and rated is None:
        rated = int(m.group(1))

    return current, rated


def _clocks_mhz(device_id=None) -> tuple[int, int]:
    idx = _npu_smi_index(device_id)
    try:
        current, rated = _parse_aicore_clocks_mhz(_npu_smi_info(idx))
    except Exception as e:
        warnings.warn(
            f"npu-smi clock query failed ({e}); using {_FALLBACK_AICORE_MHZ} MHz"
        )
        return _FALLBACK_AICORE_MHZ, _FALLBACK_AICORE_MHZ
    if rated is None:
        warnings.warn(
            f"could not parse AI Core rated freq from npu-smi; using {_FALLBACK_AICORE_MHZ} MHz"
        )
        rated = _FALLBACK_AICORE_MHZ
    if current is None:
        current = rated
    return current, rated


def get_current_gpu_clock_rate_in_khz(device_id=None):
    current_mhz, _ = _clocks_mhz(device_id)
    return current_mhz * 1000


@functools.lru_cache()
def get_max_gpu_clock_rate_in_khz(device_id=None):
    _, rated_mhz = _clocks_mhz(device_id)
    return rated_mhz * 1000


def get_aclshmem_home():
    """Directory of the importable shmem package by find_spec.

    Falls back to ACLSHMEM_HOME then the in-tree source path.
    """
    spec = importlib.util.find_spec("shmem")
    if spec and spec.origin:
        return Path(spec.origin).parent
    return Path(
        os.getenv(
            "ACLSHMEM_HOME",
            Path(__file__).parent.parent.parent
            / "3rdparty"
            / "shmem"
            / "src"
            / "python"
            / "shmem",
        )
    )


@functools.lru_cache()
def get_aclshmem_version():
    return "unknown"


@functools.lru_cache()
def get_aclshmem_lib_path():
    home = get_aclshmem_home()
    try:
        spec = importlib.util.spec_from_file_location(
            "_triton_dist_shmem_soc", home / "_soc.py"
        )
        if spec is None or spec.loader is None:
            return None
        soc_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(soc_module)
        lib_path = home / "backends" / soc_module.select_backend() / "libshmem.so"
        return lib_path if lib_path.is_file() else None
    except (AttributeError, ImportError, OSError):
        return None


@functools.lru_cache()
def get_aclshmem_hash():
    lib_path = get_aclshmem_lib_path()
    if lib_path is None:
        return "unknown"
    try:
        with lib_path.open("rb") as f:
            return hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    except OSError:
        return "unknown"


__all__ = [
    "_npu_smi_index",
    "get_max_gpu_clock_rate_in_khz",
    "get_current_gpu_clock_rate_in_khz",
    "get_aclshmem_home",
    "get_aclshmem_version",
    "get_aclshmem_lib_path",
    "get_aclshmem_hash",
]
