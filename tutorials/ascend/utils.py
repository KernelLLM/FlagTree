# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared aclshmem constants and bootstrap for the Ascend tutorials/tests.

Every tutorial under tutorials/ascend and every test under
python/triton_dist/test/ascend boots the aclshmem session through
`init_aclshmem` so the endpoint derivation (ASH_MASTER_PORT /
ASH_MASTER_ADDR / MASTER_ADDR, tcp6 for IPv6) lives in exactly one place.
The allocator ground truths (2 MB SHMEM_PAGE_SIZE, 40 GB
ACLSHMEM_MAX_LOCAL_SIZE) live here for the same reason: every pool helper
aligns to the same page size and checks the same hard cap.
Tutorials import this module by plain name — under torchrun the tutorial
directory itself is sys.path[0] — while the test prof/conftest paths put
this directory on sys.path for the tests.
"""

import os

import shmem as ash

# ACLSHMEM hands out symmetric memory in 2 MB pages and rejects heaps above
# 40 GB (ACLSHMEM_MAX_LOCAL_SIZE), so every capacity request is page-aligned
# and checked against that cap.
SHMEM_PAGE_SIZE = 2 * 1024 * 1024
ACLSHMEM_MAX_LOCAL_SIZE = 40 * 1024 * 1024 * 1024


def _get_ash_ip_port():
    ash_port = os.environ.get("ASH_MASTER_PORT", "8666")
    # Explicit ASH_MASTER_ADDR override first, else torchrun's MASTER_ADDR.
    ash_addr = os.environ.get("ASH_MASTER_ADDR") or os.environ.get(
        "MASTER_ADDR", "127.0.0.1"
    )
    if ":" in ash_addr:  # IPv6 literal needs the tcp6 scheme and brackets
        return f"tcp6://[{ash_addr}]:{ash_port}"
    return f"tcp://{ash_addr}:{ash_port}"


G_IP_PORT = _get_ash_ip_port()


def init_aclshmem(rank, world_size, local_mem_size, engine):
    """Bring up the aclshmem session on the shared endpoint.

    Args:
        rank: this process's rank.
        world_size: total number of ranks in the session.
        local_mem_size: symmetric heap size in bytes.
        engine: `ash.OpEngineType` engine the data ops should use.
    """
    if ash.set_conf_store_tls(False, "") != 0:
        raise RuntimeError("set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = local_mem_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = engine
    if ash.aclshmem_init(attributes) != 0:
        raise RuntimeError("aclshmem_init failed")
