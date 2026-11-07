"""Model-only selection of total CTA warps and their associated layouts.

Every branch starts from the same TTIR, with its own legal launch options.
Only absolute native learned scores select the winner. No GPU execution,
backend resources, compiler identity, seed delta, or prices rank branches.
"""
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import time


DEFAULT_NUM_WARPS = (1, 2, 4, 8)


def canonical_num_warps(value, initial):
    if type(initial) is not int or initial < 1 or initial > 32 or initial & (initial - 1):
        raise ValueError("initial num_warps must be a power of two between 1 and 32")
    if isinstance(value, str):
        if value == "fixed":
            return ()
        parts = value.split(",")
        if any(not p.strip().isdigit() for p in parts):
            raise ValueError("layout_num_warps requires comma-separated positive powers of two, or 'fixed'")
        value = tuple(int(p) for p in parts)
    if not isinstance(value, (tuple, list)):
        raise ValueError("layout_num_warps must be a tuple of positive powers of two")
    if not value:
        return ()
    if any(type(w) is not int or w < 1 or w > 32 or w & (w - 1) for w in value):
        raise ValueError("layout_num_warps entries must be powers of two between 1 and 32")
    return tuple(sorted(set(value) | {initial}))


def bind_num_warps(options, *, enabled):
    explicit = "layout_num_warps" in options
    value = options.get("layout_num_warps", os.environ.get("FLAGTREE_LAYOUT_TUNING_NUM_WARPS", DEFAULT_NUM_WARPS))
    if not enabled:
        if explicit and value:
            raise ValueError("layout_num_warps is supported only by the learned model in search mode")
        return {"layout_num_warps": ()} if explicit else {}
    return {"layout_num_warps": canonical_num_warps(value, options.get("num_warps", 4))}


def divide_budget(total, choices):
    if type(total) is not int or total < len(choices):
        raise ValueError("layout search budget must allow at least one trial/state per num_warps branch")
    q, r = divmod(total, len(choices))
    return {w: q + (i < r) for i, w in enumerate(choices)}


def bind_mfma_warp_domain(options, *, enabled):
    value = os.environ.get("FLAGTREE_LAYOUT_TUNING_MFMA_WARP_DOMAIN", "0")
    if value not in ("0", "1"):
        raise ValueError("MFMA warp domain must be 0 or 1")
    active = value == "1"
    if active and not enabled:
        raise ValueError("MFMA warp domain requires the learned model search pass")
    key = 'layout_mfma_warp_domain'
    if key in options and (type(options[key]) is not bool or options[key] != active):
        raise ValueError("cached MFMA warp domain differs from the explicit compile policy")
    return {key: active} if active or key in options else {}


def absolute_score(report, model_sha):
    if (report.get("active_scorer") != "generic-current-layout-v114" or
            report.get("model_active") is not True or report.get("fallback_allowed") is not False or
            report.get("selection_status") != "finite-complete-candidate"):
        raise ValueError("num_warps branch has no complete learned-model score")
    selected = report["selected_score"]
    ranking = selected["ranking"]
    if (selected.get("seed_reference_input") is not False or ranking.get("seed_reference_input") is not False or
            selected.get("fallback_allowed") is not False or ranking.get("fallback_allowed") is not False or
            ranking.get("scorer") != "generic-current-layout-v114" or
            ranking.get("deployment_sha256") != model_sha or ranking.get("units") != "ordinal-score-not-kernel-duration"):
        raise ValueError("num_warps ranking requires the same absolute learned model, without a seed delta or fallback")
    value = ranking["total"]
    if type(value) not in (float, int) or not math.isfinite(value) or not math.isclose(value, report["selected_cost"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("invalid or inconsistent num_warps score")
    return value


def choose_branch(branches, initial):
    eligible = [b for b in branches if b["status"] == "eligible"]
    if not eligible:
        reasons = "; ".join(str(b["num_warps"])+": "+b.get("reason", b["status"]) for b in branches)
        raise ValueError("model num_warps search found no legal scored candidate; no compiler/ProgramCost fallback: " + reasons)
    return min(eligible, key=lambda b: (b["score"], b["num_warps"] != initial, b["num_warps"]))


def search_num_warps(source, metadata, options, backend_cls):
    from triton._C.libtriton import ir, hcu

    if not hasattr(hcu, "run_layout_model_search"):
        raise RuntimeError("num_warps search requires the native run_layout_model_search binding")
    if not options.layout_execution_context:
        raise ValueError("num_warps search requires the actual JIT-bound execution context")
    model_path = os.environ["FLAGTREE_LAYOUT_TUNING_JOINT_MODEL"]
    model_sha = os.environ["FLAGTREE_LAYOUT_TUNING_JOINT_MODEL_SHA256"]
    choices = options.layout_num_warps
    def budget(name, default):
        value = os.environ.get("FLAGTREE_LAYOUT_TUNING_" + name, str(default))
        if not value.isdigit():
            raise ValueError("invalid layout search budget: " + name)
        return int(value)
    trials = divide_budget(budget("TRIALS", 64), choices)
    dump = os.environ.get("FLAGTREE_LAYOUT_TUNING_DUMP", "")
    trace = os.environ.get("FLAGTREE_LAYOUT_TUNING_TRACE", "")
    search_arguments = dict(
        algorithm=os.environ.get("FLAGTREE_LAYOUT_TUNING_ALGORITHM", "beam"),
        candidates=budget("CANDIDATES", 12), beam_width=budget("BEAM", 3),
        candidate_policy=os.environ.get("FLAGTREE_LAYOUT_TUNING_CANDIDATE_POLICY", "legacy"),
        candidate_pool=budget("CANDIDATE_POOL", 128),
        matrix_instruction_size=options.matrix_instr_nonkdim, kpack=options.kpack,
        mmac_layout_force=options.mmac_layout_force)
    if getattr(options, 'layout_mfma_warp_domain', False):
        search_arguments['mfma_warp_candidates'] = True
    started = time.perf_counter()
    branches = []
    payloads = {}
    # Parse separate copies of the same post-TTIR module; no blanket encoding
    # rewrite can preserve dot/slice/shared legality across total warp counts.
    with tempfile.TemporaryDirectory(prefix="triton-layout-num-warps-") as temporary:
        path = Path(temporary) / "input.ttir"
        path.write_text(str(source) + "\n")
        for warps in choices:
            row = dict(num_warps=warps, trial_budget=trials[warps], status="starting")
            begin = time.perf_counter()
            branch_options = replace(options, num_warps=warps, layout_num_warps=())
            branch_metadata = dict(metadata, num_warps=warps)
            try:
                module = ir.parse_mlir_module(str(path), source.context)
                module.context = source.context
                backend_cls.make_ttgir_layout_prefix(module, branch_options)
                report = json.loads(hcu.run_layout_model_search(
                    module, model_path, model_sha, options.layout_execution_context,
                    options.num_stages, options.arch,
                    trials=trials[warps], **search_arguments,
                    trace_path=str(Path(trace) / ("warps-" + str(warps))) if trace else ""))
                score = absolute_score(report, model_sha)
                module = backend_cls.make_ttgir_after_layout_slot(module, branch_metadata, branch_options)
                if not module.verify() or module.get_int_attr("ttg.num-warps") != warps:
                    raise ValueError("num_warps branch failed post-suffix module verification")
                row.update(status="eligible", score=score, selected_trial=report["selected_trial"],
                           trials=report["trials"], changed_values=report["changed_values"])
                # Preserve all branches, including ties. A winner-only report
                # cannot distinguish missing facts from untrained coefficients.
                # These are current-candidate facts, never new ranking inputs.
                row["scoring_evidence"] = report["selected_score"]
                if dump:
                    branch_dir = Path(dump).with_suffix(".num-warps")
                    branch_dir.mkdir(parents=True, exist_ok=True)
                    branch_path = branch_dir / ("warps-" + str(warps) + ".json")
                    branch_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
                    row["branch_report"] = str(branch_path)
                payloads[warps] = (module, branch_metadata, report)
            except (ValueError, RuntimeError) as error:
                row.update(status="rejected", reason=str(error))
            row["analysis_and_suffix_seconds"] = time.perf_counter() - begin
            branches.append(row)
    winner = choose_branch(branches, options.num_warps)
    module, selected_metadata, report = payloads[winner["num_warps"]]
    search_report = dict(policy="absolute-model-num-warps-and-layout", initial_num_warps=options.num_warps,
        candidates=list(choices), selected_num_warps=winner["num_warps"], selected_score=winner["score"],
        total_trial_budget=sum(trials.values()), trials=sum(b.get("trials", 0) for b in branches),
        branches=branches, model_sha256=model_sha, seed_delta_used=False, program_cost_used=False,
        timing_used_for_selection=False, total_analysis_and_suffix_seconds=time.perf_counter()-started)
    report["num_warps_search"] = search_report
    metadata.update(selected_metadata)
    metadata["num_warps"] = winner["num_warps"]
    metadata["layout_num_warps_search"] = search_report
    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return module
