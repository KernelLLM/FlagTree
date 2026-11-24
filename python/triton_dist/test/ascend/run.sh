#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Run every Ascend verification/performance script in this directory sequentially.
#
# The case list is discovered from the directory itself: every *.py file
# carrying a `if __name__ == "__main__"` entry is a case, in filename order
# (helper modules such as prof.py have no entry point and are skipped
# automatically).  No list to maintain when new scripts are added.
# Per-case rules:
#   - UDMA scripts (any file selecting OpEngineType.UDMA, regardless of its
#     name) require Ascend 950 and are skipped automatically otherwise.
#     Set SKIP_UDMA=1 to skip them even on 950.
#   - Autotune experiments (filename containing "autotune") are excluded by
#     default; set RUN_AUTOTUNE=1 to include them.
#   - A script failure does not stop the sweep; a summary is printed at the
#     end and the exit code is non-zero if anything failed.
#
# Usage, from this directory (two cards):
#   bash run.sh [--mode=verify|perf|both] [--case=random|full]
# Defaults: --mode=both --case=full.
#
# Environment variables:
#   NPROC_PER_NODE  processes per node            (default: 2)
#   SKIP_UDMA       set to 1 to skip UDMA cases   (default: 0)
#   RUN_AUTOTUNE    set to 1 to include autotune experiments (default: 0)
#   ASCEND_RT_VISIBLE_DEVICES  passed through, defaults to 0,1 for 2 ranks
#   ASCEND_PROFILE_WARMUP / ASCEND_PROFILE_ITERS  passed through to @prof
#   ASCEND_CASE_SEED reproducible --case=random sampling seed (default: random)
#   QKV_BENCH_S / QKV_BENCH_H / A2A_BENCH_S / A2A_BENCH_H  passed through
#
# Each run gets a fresh port via ASH_MASTER_PORT (base 31000, away from
# tutorials/ascend/run.sh's 21000 so back-to-back sweeps do not trip over
# TCP TIME_WAIT sockets).
# HCCL_NPU_SOCKET_PORT_RANGE is NOT injected: the scripts that need it set it
# themselves with os.environ.setdefault.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: bash run.sh [--mode=verify|perf|both] [--case=random|full]" >&2
}

RUN_MODE="both"
CASE_SELECTION="full"
for arg in "$@"; do
    case "${arg}" in
        --mode=*) RUN_MODE="${arg#--mode=}" ;;
        --case=*) CASE_SELECTION="${arg#--case=}" ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: ${arg}" >&2
            usage
            exit 2
            ;;
    esac
done

case "${RUN_MODE}" in
    verify|perf|both) ;;
    *)
        echo "Invalid mode: ${RUN_MODE}" >&2
        usage
        exit 2
        ;;
esac
case "${CASE_SELECTION}" in
    random|full) ;;
    *)
        echo "Invalid case selection: ${CASE_SELECTION}" >&2
        usage
        exit 2
        ;;
esac

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    ASCEND_RT_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
    export ASCEND_RT_VISIBLE_DEVICES
fi
if [ "${CASE_SELECTION}" = "random" ]; then
    ASCEND_CASE_SEED="${ASCEND_CASE_SEED:-$((RANDOM * 32768 + RANDOM))}"
    export ASCEND_CASE_SEED
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

# Match explicit UDMA names (including MoE wrappers) and engine selection
# in scripts such as fused reverse A2A whose names carry no -udma suffix.
is_udma_case() {
    case "$(basename "$1")" in
        *-udma*) return 0 ;;
    esac
    grep -q "OpEngineType.UDMA" "$1"
}

# Only torchrun entry points are cases; helper modules (prof.py) are not.
is_runnable_case() {
    grep -q '__main__' "$1"
}

run_index=0
PASS=()
FAIL=()
SKIP=()

launch() {
    local file="$1"
    run_index=$((run_index + 1))
    local port=$((31000 + 7 * run_index))
    local name
    name="$(basename "${file}" .py)"

    echo "==================================================================="
    echo "[RUN ] ${name}"
    echo "==================================================================="
    local start_ts end_ts
    start_ts=$(date +%s)
    local run_cmd=(env ASH_MASTER_PORT="${port}" \
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${file}" \
        "--mode=${RUN_MODE}" "--case=${CASE_SELECTION}")
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

# Run cases in alphabetical filename order, grouping operator variants together.
CASES=()
for file in "${SCRIPT_DIR}"/*.py; do
    name="$(basename "${file}" .py)"
    if ! is_runnable_case "${file}"; then
        continue
    fi
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
    CASES+=("${file}")
done

echo "==================================================================="
echo " Ascend test sweep"
echo " ranks=${NPROC_PER_NODE}  devices=${ASCEND_RT_VISIBLE_DEVICES}"
echo " mode=${RUN_MODE}  case=${CASE_SELECTION}  scripts=${#CASES[@]}"
if [ "${CASE_SELECTION}" = "random" ]; then
    echo " shape_sample_size=10  seed=${ASCEND_CASE_SEED}"
fi
echo " UDMA: $([ "${UDMA_OK}" = "1" ] && echo "enabled (Ascend 950)" || echo "skipped")"
echo "==================================================================="

for file in ${CASES[@]+"${CASES[@]}"}; do
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
for name in ${PASS[@]+"${PASS[@]}"}; do
    echo "  [PASS] ${name}"
done
echo "FAIL: ${#FAIL[@]}"
for name in ${FAIL[@]+"${FAIL[@]}"}; do
    echo "  [FAIL] ${name}"
done
echo "SKIP: ${#SKIP[@]}"
for name in ${SKIP[@]+"${SKIP[@]}"}; do
    echo "  [SKIP] ${name}"
done
echo "==================================================================="

if [ "${#FAIL[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
