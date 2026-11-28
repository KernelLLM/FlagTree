# Source this file in bash. Existing verified toolchain; installs nothing.
_scaledmm_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SCALEDMM_CANN_ROOT="${SCALEDMM_CANN_ROOT:-/usr/local/Ascend/cann-9.2.1}"
export SCALEDMM_VENV="${SCALEDMM_VENV:-/home/ricky/venv/triton-ascend-tla}"
export SCALEDMM_TA_ROOT="${SCALEDMM_TA_ROOT:-/home/ricky/git/Triton-Ascend-tla}"
export SCALEDMM_NPUIR_BIN="${SCALEDMM_NPUIR_BIN:-/home/ricky/git/AscendNPU-IR-tla-build/bin}"
for _scaledmm_required in "$SCALEDMM_CANN_ROOT/set_env.sh" "$SCALEDMM_VENV/bin/activate" "$SCALEDMM_TA_ROOT/python/triton/__init__.py" "$SCALEDMM_NPUIR_BIN/bishengir-compile"; do
    if [[ ! -f "$_scaledmm_required" ]]; then
        printf 'Missing required toolchain file: %s\n' "$_scaledmm_required" >&2
        return 1
    fi
done
source "$SCALEDMM_CANN_ROOT/set_env.sh" || return 1
source "$SCALEDMM_VENV/bin/activate" || return 1
export ASCEND_RT_VISIBLE_DEVICES="${SCALEDMM_DEVICE:-2}"
export OMP_NUM_THREADS=8
export TRITON_NPU_COMPILER_PATH="$SCALEDMM_NPUIR_BIN"
export BISHENG_INSTALL_PATH="$SCALEDMM_CANN_ROOT/tools/bisheng_compiler/bin"
export PATH="$SCALEDMM_NPUIR_BIN:$PATH"
export PYTHONPATH="$SCALEDMM_TA_ROOT/python"
export BISHENG_BACKEND=process
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="${SCALEDMM_CACHE_DIR:-/home/ricky/.cache/scaledmm-20261112}"
unset FLAGTREE_BUILD_DIR USE_CUSTOM_COMPILE_OPT TRITON_DEBUG TRITON_ALWAYS_COMPILE TLA_TIMING
mkdir -p -- "$TRITON_CACHE_DIR"
cd -- "$_scaledmm_root" || return 1
unset _scaledmm_root _scaledmm_required
