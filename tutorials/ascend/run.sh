#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Run every Ascend tutorial example in this directory sequentially.
#
# The case list is discovered from the directory itself: every *.py file is a
# case, in filename order.  No list to maintain when new examples are added.
# Per-case rules:
#   - UDMA cases (any file selecting OpEngineType.UDMA, regardless of its
#     name) require Ascend 950 and are skipped automatically otherwise.
#     Set SKIP_UDMA=1 to skip them even on 950.
#   - Autotune experiments (filename containing "autotune") are excluded by
#     default; set RUN_AUTOTUNE=1 to include them.
#   - A case failure does not stop the sweep; a summary is printed at the end.
#
# Usage, from the repository root (two cards):
#   bash tutorials/ascend/run.sh
#
# Environment variables:
#   NPROC_PER_NODE  processes per node            (default: 2)
#   SKIP_UDMA       set to 1 to skip UDMA cases   (default: 0)
#   RUN_AUTOTUNE    set to 1 to include autotune experiments (default: 0)
#   ASCEND_RT_VISIBLE_DEVICES  passed through, defaults to 0,1 for 2 ranks
#
# Each run gets a fresh port via ASH_MASTER_PORT so back-to-back shmem
# launches do not trip over TCP TIME_WAIT sockets.  HCCL_NPU_SOCKET_PORT_RANGE
# is NOT injected: the examples that need it set it themselves with
# os.environ.setdefault.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
    export ASCEND_RT_VISIBLE_DEVICES
fi

# ---------------------------------------------------------------------------
# Detect Ascend 950 for the UDMA cases: npu-smi reports the SoC name; fall
# back to ASCEND_SOC_VERSION when npu-smi is unavailable (e.g. containers).
# Anything unrecognized is treated as "not 950" so UDMA is skipped rather
# than launched into a failure.
# ---------------------------------------------------------------------------
is_ascend_950() {
    local soc=""
    if command -v npu-smi >/dev/null 2>&1; then
        soc="$(npu-smi info)"
    fi
    case "$soc" in
        *950*) return 0 ;;
    esac
    case "${ASCEND_SOC_VERSION:-}" in
        *950*) return 0 ;;
    esac
    return 1
}

# A case is UDMA iff its code selects the UDMA engine.  Matching the code
# (not the filename) catches cases such as reverse-a2a-norm-matmul variants whose names carry no -udma
# suffix but which do use OpEngineType.UDMA.
is_udma_case() {
    grep -q "OpEngineType.UDMA" "$1"
}

run_index=0
PASS=()
FAIL=()
SKIP=()

launch() {
    local file="$1"
    run_index=$((run_index + 1))
    local port=$((21000 + 7 * run_index))
    local name
    name="$(basename "${file}" .py)"

    echo "==================================================================="
    echo "[RUN ] ${name}"
    echo "==================================================================="
    local start_ts end_ts
    start_ts=$(date +%s)
    local run_cmd=(env ASH_MASTER_PORT="${port}" \
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${file}")
    if "${run_cmd[@]}"; then
        end_ts=$(date +%s)
        echo "[PASS] ${name} ($((end_ts - start_ts))s)"
        PASS+=("${name}")
    else
        end_ts=$(date +%s)
        echo "[FAIL] ${name} ($((end_ts - start_ts))s)"
        FAIL+=("${name}")
    fi
}

UDMA_OK=0
if [ "${SKIP_UDMA:-0}" != "1" ]; then
    if is_ascend_950; then
        UDMA_OK=1
    else
        echo "[SKIP] UDMA cases require Ascend 950; not detected on this host."
        echo "       (export ASCEND_SOC_VERSION=*950* if you are actually on 950)"
    fi
fi

echo "==================================================================="
echo " Ascend tutorial sweep"
echo " ranks=${NPROC_PER_NODE}  devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo " UDMA: $([ "${UDMA_OK}" = "1" ] && echo "enabled (Ascend 950)" || echo "skipped")"
echo "==================================================================="

# Run cases in alphabetical filename order, grouping operator variants together.
for file in "${SCRIPT_DIR}"/*.py; do
    name="$(basename "${file}" .py)"
    case "${name}" in
        *autotune*)
            if [ "${RUN_AUTOTUNE:-0}" != "1" ]; then
                SKIP+=("${name} (autotune experiment, RUN_AUTOTUNE=1 to include)")
                continue
            fi
            ;;
    esac
    if is_udma_case "${file}" && [ "${UDMA_OK}" != "1" ]; then
        SKIP+=("${name} (UDMA needs Ascend 950)")
        continue
    fi
    launch "${file}"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==================================================================="
echo " Summary"
echo "==================================================================="
echo "PASS: ${#PASS[@]}"
for name in "${PASS[@]}"; do
    echo "  [PASS] ${name}"
done
echo "FAIL: ${#FAIL[@]}"
for name in "${FAIL[@]}"; do
    echo "  [FAIL] ${name}"
done
echo "SKIP: ${#SKIP[@]}"
for name in "${SKIP[@]}"; do
    echo "  [SKIP] ${name}"
done
echo "==================================================================="

if [ "${#FAIL[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
