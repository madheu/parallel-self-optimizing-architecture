# PSOA A/B Benchmark Report

**Model:** agnes-2.0-flash | **Temperature:** 0.3 | **PSOA Max Rounds:** 3

## Overall

| Metric | Direct (A) | PSOA (B) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 17/20 (85.0%) | 18/20 (90.0%) | **+5.0%** |
| **Total Time** | 400.9s | 2882.7s | +2481.9s |
| **Total Tokens** | 12,997 | 212,357 | +199,360 |
| **Convergence Rate** | - | 19/20 (95.0%) | - |
| **Avg PSOA Rounds** | - | 1.35 | - |

## By Category

| Category | Count | Direct | PSOA | Delta |
| :--- | :---: | :---: | :---: | :---: |
| 数学 | 6 | 83.3% | 100.0% | +16.7% |
| 逻辑 | 7 | 71.4% | 85.7% | +14.3% |
| 常识 | 7 | 100.0% | 85.7% | -14.3% |

## Per Question

| # | Cat | Direct | PSOA | D.Time | P.Time | Rounds | Conv |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 数学 | OK | OK | 39.4s | 43.9s | 1 | Y |
| 2 | 数学 | OK | OK | 7.8s | 88.3s | 1 | Y |
| 3 | 数学 | OK | OK | 10.6s | 68.7s | 1 | Y |
| 4 | 数学 | NO | OK | 54.1s | 320.3s | 2 | Y |
| 5 | 数学 | OK | OK | 27.1s | 74.1s | 1 | Y |
| 6 | 数学 | OK | OK | 17.0s | 83.1s | 1 | Y |
| 7 | 逻辑 | OK | OK | 17.9s | 132.3s | 2 | Y |
| 8 | 逻辑 | OK | OK | 7.9s | 255.7s | 2 | Y |
| 9 | 逻辑 | NO | OK | 10.6s | 165.1s | 1 | Y |
| 10 | 逻辑 | OK | OK | 10.7s | 90.7s | 1 | Y |
| 11 | 逻辑 | OK | OK | 9.4s | 127.6s | 1 | Y |
| 12 | 逻辑 | OK | NO | 21.8s | 140.3s | 1 | Y |
| 13 | 逻辑 | NO | OK | 9.0s | 362.4s | 3 | Y |
| 14 | 常识 | OK | OK | 7.6s | 85.1s | 1 | Y |
| 15 | 常识 | OK | OK | 44.4s | 77.5s | 1 | Y |
| 16 | 常识 | OK | NO | 39.9s | 313.3s | 3 | N |
| 17 | 常识 | OK | OK | 9.8s | 79.0s | 1 | Y |
| 18 | 常识 | OK | OK | 30.6s | 74.1s | 1 | Y |
| 19 | 常识 | OK | OK | 8.3s | 211.3s | 1 | Y |
| 20 | 常识 | OK | OK | 16.9s | 90.0s | 1 | Y |

---
*PSOA accuracy improvement: +5.0%, extra time: 2482s, extra token cost ~$0.42, convergence: 95.0%*

## Anomaly Analysis

### Type A: Converged but Wrong (L false positive)

| # | Cat | Rounds | D.Time | P.Time |
| :--- | :--- | :---: | :---: | :---: |
| 12 | 逻辑 | 1 | 21.8s | 140.3s |

### Type B: Correct but Not Converged (R false negative)

None


### Type C: Regression (Direct OK, PSOA NO)

PSOA introduced 2 regression(s): 
- **Q12** [逻辑]: 甲说乙说谎，乙说丙说谎，丙说甲乙都说谎。谁说真话？
- **Q16** [常识]: 老人生于1900年2月29日，到2020年过了多少个真正生日（2月29日）？

### Type D: Fix (Direct NO, PSOA OK)

PSOA fixed 3 question(s):
- **Q4** [数学]: 一张桌子的高度：站猫-蹲龟=150cm，蹲猫-站龟=110cm。求桌高。
- **Q9** [逻辑]: 狼、羊、白菜过河，船每次带一样。如何安全过河？
- **Q13** [逻辑]: 5人戴红蓝帽子，从后往前猜自己颜色，可以说红/蓝/过。如何保证至少4人对？

### Summary

- **L false positive rate**: 1/19 converged results are wrong
- **R false negative rate**: 0/20 PSOA runs failed to converge despite correct answer
- **Net improvement**: 3 fixes - 2 regressions = 1 net gain
