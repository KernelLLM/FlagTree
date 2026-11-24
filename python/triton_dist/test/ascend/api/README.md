# Ascend API Tests

This directory contains API tests for Triton-distributed-ascend, covering symmetric memory addressing, data transfer, signal synchronization, barriers, and autotuning. The tests use pytest to verify interface behavior and result correctness.

## Prerequisites

- Complete [Installation](../../../../../docs_ascend/en/getting-started/installation.md) and [Build from Source](../../../../../docs_ascend/en/getting-started/build.md). Ensure that pytest, PyTorch, torch_npu, Triton, `triton_dist`, and shmem are installed in the current Python environment.
- Load the CANN environment variables in the current shell and configure the AscendNPU-IR tool paths as described in the build guide.
- Most tests that use multiple NPUs require 2 devices; the subgroup barrier test requires 4. At least 4 visible NPUs are needed to run the test selection below.
- UDMA runtime tests also require Ascend 950PR&950DT products with UDMA support and the corresponding software environment.

## Run the tests

Run the following command from the repository root:

```bash
pytest python/triton_dist/test/ascend/api -m dist
```

`-m dist` selects only tests marked `dist`; it does not run every test in this directory. For tests that use multiple NPUs, the test framework creates worker processes and configures ranks automatically. No additional `torchrun` command is needed.

If only 2 NPUs are available, start with the wait and notify interface tests:

```bash
pytest python/triton_dist/test/ascend/api/test_wait_notify.py -m dist
```

## Read the results

pytest summarizes the results at the end: `passed` indicates success, `failed` indicates an assertion or execution failure, `error` indicates a collection, setup, or teardown error, `skipped` indicates a skipped test, and `deselected` indicates a test excluded by the selection criteria. Skipped or deselected tests have not passed validation.

For failures, use the test name, traceback, and rank information in the pytest output to locate the problem.

## Transport runtime coverage

The `dist` selection includes AUTO, MTE, and UDMA runtime tests. RoCE runtime tests intentionally have no `dist` mark, pending official enablement; invoke their individual cases explicitly only in a configured RoCE environment.

| API family | Runtime test | Coverage |
|---|---|---|
| Grouped atomics and compare-swap | `test_atomic_transport_runtime.py` | All supported operation selectors for AUTO/MTE/UDMA; returned values and remote updates |
| Low-level get/put and quiet | `test_rma_transport_runtime.py` | MTE, ordinary UDMA, and UDMA QPs 0/1; typed offsets, byte counts, and destination guards |
| `putmem_signal_nbi_lowlevel` | `test_rma_transport_runtime.py` | UDMA (default and only supported transport), ordinary and QP variants; payload and uint64 notification values |
| `sync` / `sync_all` with UDMA initialized | `test_sync_ops.py` | Explicit UDMA put + quiet followed by generic team/world synchronization across repeated epochs |
| Relay get/put and quiet | `test_rma_transport_runtime.py` | Distinct source, destination, and relay PEs; requires four visible NPUs; no `dist`/`ci` marks |
| RoCE atomics, RMA, barriers and sync | `test_atomic_transport_runtime.py`, `test_rma_transport_runtime.py`, `test_roce_collective_runtime.py` | Runtime cases provided, excluded from `dist` pending official enablement |

UDMA QP tests configure two QPs through `aclshmemx_set_qp_num` before SHMEM initialization and require a direct-mode SHMEM/template build. Ordinary UDMA and QP tests run by default on Ascend950.

`sync` and `sync_all` with the default `transport=AUTO` work when UDMA is initialized, but their generic implementation synchronizes through shared-memory signaling. They do not submit UDMA synchronization requests or drain outstanding UDMA queues. Complete those requests with `quiet_lowlevel(peer, transport=ACLSHMEM_TRANSPORT_UDMA)` before the collective. Explicit `transport=UDMA` remains rejected because SHMEM provides no UDMA-specific sync entry point.

Atomic dispatch checks `is_compile_on_910_95()`. Before Ascend950, only AUTO/MTE `atomic_add` on int8, int16, int32, float16, bfloat16 and float32 is supported; other atomic operation/type/transport combinations require Ascend950 or later. ABI tests model an Ascend950 target explicitly, with separate tests for pre-950 rejection.

All frontend dispatch and lowering tests, including RoCE and relay, carry both `ci` and `dist` marks and require no hardware.

Relay runtime tests intentionally have neither `dist` nor `ci` marks because the default NPU-IR package is built without relay support. They require SHMEM built with `-enable_relay` and compiler templates built with `BISHENGIR_SHMEM_RELAY_SUPPORT=ON`. For that environment, run:

```bash
TRITON_DIST_TEST_UDMA_RELAY=1 pytest python/triton_dist/test/ascend/api/test_rma_relay.py python/triton_dist/test/ascend/api/test_rma_transport_runtime.py -k relay
```

This flag declares the installed build configuration because SHMEM has no Python relay-capability query. It allows explicitly selected relay runtime cases to run and skips incompatible UDMA QP cases. It does not add any test marks. Without it, relay runtime cases are skipped. Runtime tests also skip when the required NPUs or SoC are unavailable. A skip is not runtime validation.
