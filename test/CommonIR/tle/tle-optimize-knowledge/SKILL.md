---
name: tle-optimize-knowledge
description: >-
  Reference-only knowledge base for TLE DSA Ascend kernel optimization. Read this when
  a kernel is correct but not fast enough, and you need to identify the right optimization
  direction before changing code. Does not define the optimization workflow itself.
---

# TLE Optimize Knowledge — Ascend DSA

## Purpose

This skill is a routing aid for TLE DSA kernel performance work. It holds pattern cards that map common performance symptoms to optimization actions. It does not own round artifacts, profiling data, or optimization workflow orchestration.

## Scope

- This skill is reference-only.
- It does not define optimization workflow, validation rules, or round tracking.
- It does not own profiling results or performance comparison data.

## Reading Order

1. Read `references/pattern_index.md` to identify which pattern applies to the current bottleneck.
2. Read only the one or two most relevant detailed pattern cards.
3. Return to the optimization workflow with a specific, evidence-backed action.

## Relationship to other skills

- `tle-commonir-patterns` owns correctness patterns — fix correctness issues there first.
- `tle-repair-guide` owns bug fixes — a wrong result is not a performance problem.
- This skill owns optimization knowledge only — do not apply these patterns to a kernel that is not yet correct.
