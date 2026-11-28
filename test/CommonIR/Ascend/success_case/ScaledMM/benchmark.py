"""Standalone Scaled MM correctness and NPU Graph comparison (Ascend 950)."""
import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def graph_time(torch, fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(10):
            fn()
    for _ in range(3):
        graph.replay()
    torch.npu.synchronize()
    samples = []
    for _ in range(5):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        # 100 calls: milliseconds * 1000 / 100 = microseconds per call.
        samples.append(begin.elapsed_time(end) * 10)
    return statistics.median(samples)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=["original", "tla", "ascendc"], default=["original", "tla", "ascendc"])
    parser.add_argument("--m", nargs="+", type=int, choices=[32, 128, 1024], default=[32, 128, 1024])
    parser.add_argument("--dtype", nargs="+", choices=["fp16", "bf16"], default=["fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    if not args.check_only and os.environ.get("TRITON_DEBUG", "0") not in ("0", "", "false", "False"):
        parser.error("Unset TRITON_DEBUG before performance measurement.")

    import torch
    import torch_npu
    import triton
    if not torch.npu.is_available():
        raise RuntimeError("No usable NPU in this environment")
    modules = {}
    if "original" in args.methods:
        modules["original"] = importlib.import_module("scaled_mm_original")
    if "tla" in args.methods:
        modules["tla"] = importlib.import_module("scaled_mm_tla")
    if "ascendc" in args.methods and not hasattr(torch_npu, "npu_fused_matmul"):
        raise RuntimeError("torch_npu.npu_fused_matmul is required for the official composite")
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version, "python_executable": sys.executable,
            "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "triton": triton.__version__, "triton_file": triton.__file__,
            "device_name": torch.npu.get_device_name(0),
            "variables": {k: os.environ.get(k) for k in ["ASCEND_RT_VISIBLE_DEVICES", "TRITON_NPU_COMPILER_PATH", "BISHENG_INSTALL_PATH", "BISHENG_BACKEND", "TRITON_CACHE_DIR"]},
        },
        "definitions": {
            "original": "Unmodified pinned FlagGems-vllm triton_scaled_mm",
            "tla": "Triton Ascend TLA rewrite with 3 measured specializations",
            "ascendc": "Official composite: fused_matmul(16cast32) + FP32 row mul + FP32 column mul + cast; four kernels",
            "reference": "FP32 matmul * (FP32 row_scale * FP32 column_scale), cast to output dtype",
            "tolerance": {"atol": 0.01, "rtol": 0.01},
            "timing": "NPU Graph, 10 captured calls, 10 replays, median of 5; repeat in reversed method order; median of both. JIT/host time excluded. Output NaN fill excluded.",
            "speedup": "baseline microseconds / TLA microseconds; >1 means TLA faster",
            "input": "A/B = randn * 0.1; FP32 row/column scales = rand + 0.5; N=K=4096; seed=base_seed+M",
        },
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "scaled_mm_original.py", ROOT / "scaled_mm_tla.py", Path(__file__)]},
        "rows": [],
    }
    print(json.dumps(document["environment"], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    any_failure = False
    prior_timing = os.environ.pop("TLA_TIMING", None)
    try:
        for dtype_name in args.dtype:
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
            for m in args.m:
                torch.manual_seed(args.seed + m)
                a = torch.randn((m, 4096), device="npu", dtype=dtype) * 0.1
                b = torch.randn((4096, 4096), device="npu", dtype=dtype) * 0.1
                sa = torch.rand(m, device="npu", dtype=torch.float32) + 0.5
                sb = torch.rand(4096, device="npu", dtype=torch.float32) + 0.5
                reference = ((a.float() @ b.float()) * (sa[:, None] * sb[None, :])).to(dtype)
                def official():
                    acc = torch_npu.npu_fused_matmul(a, b, fused_op_type="16cast32")
                    return (acc * sa[:, None] * sb[None, :]).to(dtype)
                fns = {}
                for name in args.methods:
                    if name == "original":
                        fns[name] = lambda: modules["original"].triton_scaled_mm(a, b, sa, sb, dtype)
                    elif name == "tla":
                        fns[name] = lambda: modules["tla"].scaled_mm(a, b, sa, sb, dtype)
                    else:
                        fns[name] = official
                row = {"m": m, "n": 4096, "k": 4096, "dtype": dtype_name, "seed": args.seed + m, "precision": {}, "us": {}}
                valid = {}
                for name, fn in fns.items():
                    try:
                        output = fn()  # The unchanged TLA wrapper fills output with NaN here.
                        torch.npu.synchronize()
                        torch.testing.assert_close(output.float(), reference.float(), atol=0.01, rtol=0.01)
                        row["precision"][name] = {"status": "pass", "max_abs": (output.float() - reference.float()).abs().max().item()}
                        valid[name] = fn
                    except Exception as exc:
                        any_failure = True
                        row["precision"][name] = {"status": "fail", "error": repr(exc)}
                if not args.check_only:
                    os.environ["TLA_TIMING"] = "1"
                    samples = {name: [] for name in valid}
                    try:
                        for order in [list(valid), list(valid)[::-1]]:
                            for name in order:
                                samples[name].append(graph_time(torch, valid[name]))
                    finally:
                        os.environ.pop("TLA_TIMING", None)
                    row["samples_us"] = samples
                    row["us"] = {name: statistics.median(values) for name, values in samples.items()}
                    if "tla" in row["us"]:
                        row["tla_speedup"] = {name: value / row["us"]["tla"] for name, value in row["us"].items() if name != "tla"}
                document["rows"].append(row)
                args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if prior_timing is None:
            os.environ.pop("TLA_TIMING", None)
        else:
            os.environ["TLA_TIMING"] = prior_timing
    print("COMPLETE: " + str(args.output), flush=True)
    return 1 if any_failure else 0

if __name__ == "__main__":
    sys.exit(main())
