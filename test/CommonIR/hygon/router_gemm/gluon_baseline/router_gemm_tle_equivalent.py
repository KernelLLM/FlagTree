"""Canonical entry point for the nine audited TLE-equivalent Gluon specializations.

The exporter may specialize layouts from both launch config and static shape.
Keeping the exact source selected for every official M avoids synthesizing a
new layout expression during archival.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SOURCES = {
    1: "router-gemm-m1-n256-k7168-e9e4c0f8d9ca-dac291f31ab8a83f.py",
    8: "router-gemm-m8-n256-k7168-563ef27b411c-ffbf7b2a1b396975.py",
    16: "router-gemm-m16-n256-k7168-91efb3819721-409a50ed425e3c59.py",
    32: "router-gemm-m32-n256-k7168-25ec8b6abf8f-f9642f52c37804a7.py",
    64: "router-gemm-m64-n256-k7168-ff9727c42e3a-52008239d1c1b92a.py",
    128: "router-gemm-m128-n256-k7168-d6be54e7a96e-290e439708e84ea7.py",
    256: "router-gemm-m256-n256-k7168-a6f349652ff2-290e439708e84ea7.py",
    512: "router-gemm-m512-n256-k7168-d520122fb141-290e439708e84ea7.py",
    1024: "router-gemm-m1024-n256-k7168-61c0e18a23da-b1e042afc6e9b312.py",
}


def source_for_m(m):
    key = int(m)
    try:
        return ROOT / SOURCES[key]
    except KeyError as error:
        raise ValueError(("unsupported official router_gemm M", key)) from error


def load_kernel(m):
    path = source_for_m(m)
    name = "_router_gemm_tle_equivalent_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.KERNEL
