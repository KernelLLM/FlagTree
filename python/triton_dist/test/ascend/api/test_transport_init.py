# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest

from triton_dist.language.extra import libshmem_device as shmem
from test_atomic_transport_runtime import G_ASH_SIZE, _init_aclshmem

pytestmark = [pytest.mark.ci, pytest.mark.dist]


@pytest.mark.parametrize(
    "transport,logic,qp_count,engines,completion",
    [
        (
            shmem.ACLSHMEM_TRANSPORT_AUTO,
            False,
            None,
            ("MTE",),
            shmem.ACLSHMEM_TRANSPORT_AUTO,
        ),
        (
            shmem.ACLSHMEM_TRANSPORT_MTE,
            False,
            None,
            ("MTE",),
            shmem.ACLSHMEM_TRANSPORT_MTE,
        ),
        (
            shmem.ACLSHMEM_TRANSPORT_AUTO,
            True,
            None,
            ("MTE", "UDMA"),
            shmem.ACLSHMEM_TRANSPORT_UDMA,
        ),
        (
            shmem.ACLSHMEM_TRANSPORT_UDMA,
            False,
            None,
            ("MTE", "UDMA"),
            shmem.ACLSHMEM_TRANSPORT_UDMA,
        ),
        (
            shmem.ACLSHMEM_TRANSPORT_UDMA,
            False,
            2,
            ("MTE", "UDMA"),
            shmem.ACLSHMEM_TRANSPORT_UDMA,
        ),
        (
            shmem.ACLSHMEM_TRANSPORT_ROCE,
            False,
            2,
            ("MTE", "ROCE"),
            shmem.ACLSHMEM_TRANSPORT_ROCE,
        ),
    ],
)
def test_transport_bootstrap_delegates_after_qp_configuration(
    monkeypatch, transport, logic, qp_count, engines, completion
):
    ash = pytest.importorskip("shmem")
    import utils

    calls = []

    def configure_qps(engine, count):
        calls.append(("qp", engine, count))
        return 0

    def initialize(rank, world_size, heap, engine):
        calls.append(("init", rank, world_size, heap, engine))

    monkeypatch.setattr(ash, "aclshmemx_set_qp_num", configure_qps)
    monkeypatch.setattr(utils, "init_aclshmem", initialize)
    runtime, actual_completion = _init_aclshmem(
        1, 4, transport, logic=logic, qp_count=qp_count
    )
    engine_bits = 0
    for name in engines:
        engine_bits |= getattr(ash.OpEngineType, name).value
    expected = []
    if qp_count is not None:
        expected.append(("qp", getattr(ash.OpEngineType, engines[-1]), qp_count))
    expected.append(("init", 1, 4, G_ASH_SIZE, ash.OpEngineType(engine_bits)))
    assert calls == expected
    assert runtime is ash
    assert actual_completion == completion


@pytest.mark.parametrize(
    "master,override,expected",
    [
        ("192.0.2.10", None, "tcp://192.0.2.10:29990"),
        ("2001:db8::10", None, "tcp6://[2001:db8::10]:29990"),
        ("192.0.2.10", "2001:db8::20", "tcp6://[2001:db8::20]:29990"),
    ],
)
def test_shared_bootstrap_endpoint(monkeypatch, master, override, expected):
    pytest.importorskip("shmem")
    import utils

    monkeypatch.setenv("MASTER_ADDR", master)
    monkeypatch.setenv("ASH_MASTER_PORT", "29990")
    monkeypatch.delenv("ASH_MASTER_ADDR", raising=False)
    if override is not None:
        monkeypatch.setenv("ASH_MASTER_ADDR", override)
    assert utils._get_ash_ip_port() == expected
