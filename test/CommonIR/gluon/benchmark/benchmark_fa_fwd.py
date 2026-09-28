"""FP16 FA forward: existing Gluon implementation vs MetaX-MACA/flashattn.

Example (select an idle GPU first)::

    MACA_VISIBLE_DEVICES=7 python benchmark_fa_fwd.py

Only Sk varies by default: B=1, H=32, Sq=1024, D=128, non-causal.
Use --seq-len 2048 4096 8192 16384 32768 to sweep Sq=Sk together.
Use --head-dim 64 with the automatic-layout kernel2 implementation for D64.
For long sequences, --layout-timeout-seconds can extend candidate tuning time.
Shapes in reports are BxHxSqxSkxD. Speedup = flash_attn_ms / gluon_ms.
Inputs use each implementation's native contiguous layout (BHSD / BSHD).
GPU graph timing excludes input conversion, compilation and Python dispatch;
it includes all GPU work in each forward call, with no L2 cache flush.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
from datetime import datetime, timezone

import torch
import triton
from triton.testing import do_bench_cudagraph
from flash_attn.flash_attn_interface import _flash_attn_forward


HERE = Path(__file__).resolve().parent
DEFAULT_IMPLEMENTATION = HERE.parent / "10-maca-flash-fwd-fused-single-kernel.py"
DEFAULT_SK = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
ATOL, RTOL = 2e-3, 1e-2
COLUMNS = ("shape_BxHxSqxSkxD", "dtype", "causal", "gluon_ms", "flash_attn_ms",
           "speedup", "accuracy_ok")


def load_implementation(path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("gluon_fa_fwd_benchmark_impl", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_layout_timeout(seconds):
    if seconds is None:
        return
    # This FlagTree runtime has a fixed candidate timeout, imported by its
    # parallel tuner. Override both bindings for this benchmark process only.
    from triton.experimental.gluon import _layout_autotune, _parallel_compile_autotuner

    _layout_autotune._AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS = seconds
    _parallel_compile_autotuner._AUTOTUNE_CANDIDATE_TIMEOUT_SECONDS = seconds
    # Keep the tuning cache identity consistent with the actual protocol.
    _layout_autotune._MEASUREMENT_PROTOCOL["candidate_timeout_seconds"] = seconds


def check_tensor(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    # All-masked causal rows can have +inf LSE in both implementations.
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    difference = (actual.float()[finite] - expected.float()[finite]).abs()
    return difference.max().item() if difference.numel() else 0.0


def benchmark_case(args, implementation, sq, sk):
    torch.manual_seed(args.seed)
    q = torch.randn((args.batch, args.heads, sq, args.head_dim), device="cuda", dtype=torch.float16)
    k = torch.randn((args.batch, args.heads, sk, args.head_dim), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    # Copies and output allocation are outside the timed regions.
    q_fa, k_fa, v_fa = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
    out = torch.empty_like(q)
    lse = torch.empty((args.batch, args.heads, sq), device="cuda", dtype=torch.float32)
    scale = args.head_dim ** -0.5

    def gluon_forward():
        return implementation.launch_flash_fwd(q, k, v, out=out, lse=lse, scale=scale, causal=args.causal)

    def flash_forward():
        return _flash_attn_forward(q_fa, k_fa, v_fa, 0.0, scale, args.causal, (-1, -1))

    # Complete JIT/layout selection and correctness checks before graph capture.
    gluon_forward()
    flash_result = flash_forward()
    torch.cuda.synchronize()
    errors = {
        "output_max_abs_error": check_tensor(out, flash_result[0].transpose(1, 2)),
        "lse_max_abs_error": check_tensor(lse, flash_result[5]),
    }
    del flash_result
    functions = {"gluon": gluon_forward, "flash_attn": flash_forward}
    for _ in range(args.warmup):
        gluon_forward()
        flash_forward()
    torch.cuda.synchronize()

    samples = {name: [] for name in functions}
    for round_index in range(args.rounds):
        order = ("gluon", "flash_attn") if round_index % 2 == 0 else ("flash_attn", "gluon")
        for name in order:
            samples[name].append(do_bench_cudagraph(functions[name], rep=args.rep_ms, return_mode="median"))
    gluon_ms, flash_ms = (statistics.median(samples[name]) for name in ("gluon", "flash_attn"))
    return {
        "shape_BxHxSqxSkxD": f"{args.batch}x{args.heads}x{sq}x{sk}x{args.head_dim}",
        "dtype": "fp16", "causal": args.causal,
        "gluon_ms": gluon_ms, "flash_attn_ms": flash_ms,
        "speedup": flash_ms / gluon_ms, "accuracy_ok": True,
        **errors, "round_medians_ms": samples,
    }


def table_row(row):
    values = [f"{row[key]:.6f}" if key.endswith("_ms") else
              f"{row[key]:.3f}" if key == "speedup" else str(row[key]) for key in COLUMNS]
    return "| " + " | ".join(values) + " |"


def write_results(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["results"])
    parameters = report["parameters"]
    equal_lengths = parameters["seq_len"] is not None
    sweep = "Sq=Sk=S sweep" if equal_lengths else "Sk sweep"
    head_dim = parameters["head_dim"]
    lines = [
        f"# FP16 FlashAttention forward: {sweep}", "",
        f"GPU: {report['gpu']}; MACA_VISIBLE_DEVICES={report['maca_visible_devices']}; {report['date_utc']}.",
        f"Gluon source: `{report['implementation']}`.",
        f"Baseline: [MetaX-MACA/flashattn](https://github.com/MetaX-MACA/flashattn), installed wheel `{report['flash_attn']}`.",
        "The installed wheel's source commit is unknown; package origin is recorded in JSON.",
        f"FP16; D={head_dim}; dropout=0; causal={parameters['causal']}; scale=1/sqrt({head_dim}).",
        "Shape = B x H x Sq x Sk x D. Q/K/V have the same number of heads.",
        ("Sq=Sk=S varies; B/H/D stay fixed." if equal_lengths else "Only Sk varies; B/H/Sq/D stay fixed."),
        "Gluon uses contiguous BHSD, flash_attn uses contiguous BSHD with the same values.",
        "GPU graph timing covers all device work in forward (including O/LSE), excluding compilation,",
        "layout tuning, input conversion and Python dispatch. No L2 flush; this is steady-state GPU latency.",
        f"Layout candidate timeout: {parameters.get('layout_timeout_seconds') or 'runtime default'} seconds.",
        f"{report['parameters']['warmup']} warmup calls per provider; {report['parameters']['rounds']} alternating rounds; "
        f"{report['parameters']['rep_ms']} ms target per graph; median of round medians (10 replays per round).",
        f"Output and LSE checked for every shape: atol={ATOL}, rtol={RTOL}; errors and samples are in JSON.",
        "Speedup = flash_attn_ms / gluon_ms; >1 means Gluon is faster.", "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
        *(table_row(row) for row in report["results"]), "", "Reproduce:", "", "```bash",
        report["command"], "```", "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--sq", type=int, help="fixed query length for --sk (default: 1024)")
    lengths = parser.add_mutually_exclusive_group()
    lengths.add_argument("--sk", type=int, nargs="+", help=f"key lengths (default: {DEFAULT_SK})")
    lengths.add_argument("--seq-len", type=int, nargs="+", help="self-attention lengths: Sq=Sk=S")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--implementation", type=Path, default=DEFAULT_IMPLEMENTATION)
    parser.add_argument("--layout-timeout-seconds", type=int,
                        help="override the FlagTree layout candidate timeout for long sequences")
    parser.add_argument("--warmup", type=int, default=10, help="warmup calls per provider")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rep-ms", type=int, default=20, help="target graph duration per timing round")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "fa_fwd_fp16_sk", help="output stem for CSV/JSON/Markdown")
    args = parser.parse_args()
    if args.seq_len is not None:
        if args.sq is not None:
            parser.error("--seq-len sets both Sq and Sk; do not combine it with --sq")
        args.seq_len = sorted(set(args.seq_len))
        cases = [(s, s) for s in args.seq_len]
    else:
        args.sq = 1024 if args.sq is None else args.sq
        args.sk = sorted(set(DEFAULT_SK if args.sk is None else args.sk))
        cases = [(args.sq, sk) for sk in args.sk]
    if min(args.batch, args.heads, *(s for case in cases for s in case), args.rounds, args.rep_ms) <= 0 or args.warmup < 0:
        parser.error("shape sizes, rounds and rep-ms must be positive; warmup must be nonnegative")
    if args.layout_timeout_seconds is not None and args.layout_timeout_seconds <= 0:
        parser.error("layout-timeout-seconds must be positive")
    args.implementation = args.implementation.resolve()
    implementation = load_implementation(args.implementation)
    supported_dims = getattr(implementation, "SUPPORTED_HEAD_DIMS", (implementation.D,))
    if args.head_dim not in supported_dims:
        parser.error(f"{args.implementation.name} supports head dimensions {supported_dims}; use kernel2 for D64")
    configure_layout_timeout(args.layout_timeout_seconds)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; select a working MetaX GPU with MACA_VISIBLE_DEVICES")
    distribution = importlib.metadata.distribution("flash-attn")
    visible = os.environ.get("MACA_VISIBLE_DEVICES", "unset")
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    if visible != "unset":
        command = f"MACA_VISIBLE_DEVICES={shlex.quote(visible)} " + command
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "maca_visible_devices": visible,
        "torch": torch.__version__, "triton": triton.__version__, "flash_attn": distribution.version,
        "flash_attn_origin": json.loads(distribution.read_text("direct_url.json") or "null"),
        "implementation": str(args.implementation),
        "implementation_sha256": hashlib.sha256(args.implementation.read_bytes()).hexdigest(),
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "command": command, "results": [],
    }
    print(f"GPU: {report['gpu']}; MACA_VISIBLE_DEVICES={visible}; flash-attn={distribution.version}", flush=True)
    print(f"Gluon: {args.implementation.name}; FP16; D={args.head_dim}; causal={args.causal}", flush=True)
    print("Speedup = flash_attn_ms / gluon_ms (>1 means Gluon is faster)", flush=True)
    print("| " + " | ".join(COLUMNS) + " |", flush=True)
    print("| " + " | ".join("---" for _ in COLUMNS) + " |", flush=True)
    with torch.inference_mode():
        for sq, sk in cases:
            row = benchmark_case(args, implementation, sq, sk)
            report["results"].append(row)
            write_results(args.output, report)
            print(table_row(row), flush=True)
    print(f"Saved: {args.output.with_suffix('.csv')}, .json, .md", flush=True)


if __name__ == "__main__":
    main()
