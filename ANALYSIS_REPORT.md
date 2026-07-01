# PSOA Benchmark Analysis Report

> A/B comparison: Direct Answer vs PSOA Architecture (v2)
> Model: agnes-2.0-flash | Temperature: 0.3 | PSOA Max Rounds: 3

## Summary

| Metric | Direct (A) | PSOA (B) | Delta |
|--------|-----------|----------|-------|
| Accuracy | 80-85% (16-17/20) | 85% (17/20) | +0~5% |
| Avg Time/Question | ~20s | ~170s | +8.5x |
| Avg PSOA Rounds | - | 1.55 | - |
| Convergence Rate | - | 80% | - |

## Key Anomalies

### Type A: L False Positive (Converged but Wrong)
- **Q12** (逻辑): Direct OK, PSOA NO, conv=1r — L endorsed wrong answer
- **Q16** (常识): Direct OK, PSOA NO, conv=3r — L endorsed wrong answer after 3 rounds
- **Implication**: Convergence != Correctness. The L thread lacks external verification.

### Type B: R False Negative (Correct but Not Converged)
- **Q4** (数学): Direct NO, PSOA OK, NC(3r) — R kept flagging fatal errors despite correct answer
- **Q9** (逻辑): Direct NO, PSOA OK, NC(3r) — Same pattern
- **Implication**: R is too aggressive on certain question types. The `has_fatal_error()` heuristic (which force-promotes certain error types to fatal) may be causing unnecessary rejections.

### Type C: Regression
- Q10 (逻辑硬币题): Direct OK → PSOA NO
- Q12 (谁说真话): Direct OK → PSOA NO
- Q16 (闰年生日): Direct OK → PSOA NO

### Type D: Fixes
- Q4 (cat/turtle/table): Direct NO → PSOA OK (3 rounds, no convergence but correct)
- Q9 (wolf/goat/cabbage): Direct NO → PSOA OK
- Q11 (liar paradox): Direct NO → PSOA OK (1 round!)
- Q20 (bacteria division): Direct NO → PSOA OK (1 round)

## Verifier Priority

The most critical finding: **L false positives (Q12, Q16)** and **R false negatives (Q4, Q9)** together mean the G→R→L→I loop's convergence signal has weak correlation with answer quality. 

A V-thread (Verifier) would:
1. Independently grade the final answer against the question BEFORE marking convergence
2. Break the circular dependency between R (reviews G) and L (reviews G, referencing R)
3. Provide clean training signal: converged+certified vs converged+rejected trajectories

## Files

- `benchmark.py` — Full A/B benchmark framework (20 questions, scoring, report, anomaly analysis)
- `psoa_demo_v2.py` — PSOA v2 core with `run_psoa_single()` export
- `.reasonix/autoresearch/` — Task state and iteration logs
