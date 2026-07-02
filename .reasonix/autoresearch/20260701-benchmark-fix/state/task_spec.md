# Task Spec: PSOA Benchmark Data Fix & Analysis

## Goal
Fix data gaps in benchmark.py (token tracking, direct timing), re-run to get complete cost/benefit data, analyze R/L convergence anomalies.

## Scope
- Fix benchmark.py: track direct_tokens, psoa_tokens, direct_time properly
- Re-run benchmark with complete data
- Analyze Q12/Q16 (converged but wrong) and Q4/Q9 (correct but not converged)
- Propose V-thread (Verifier) improvement based on evidence

## Non-Goals
- Not rewriting PSOA core architecture
- Not deploying anything

## Success Criteria
1. benchmark_report.md has non-zero Direct Time and Token columns
2. Re-run confirms the anomaly pattern
3. Anomaly analysis written to .reasonix/autoresearch/20260701-benchmark-fix/

## Verification Gates
- benchmark.py import passes syntax check
- Report columns are populated
- Anomaly findings file exists
