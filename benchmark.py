#!/usr/bin/env python3
"""
PSOA A/B Benchmark
==================
对比"直接回答" vs "PSOA架构"在 20 道题上的准确率、耗时、成本。
"""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import AsyncOpenAI, APIError

# 导入 PSOA 核心
from psoa_demo_v2 import Config as PSOAConfig, run_psoa_single


# ==================== 配置 ====================

API_KEY = os.environ.get("AGNES_API_KEY")
if not API_KEY:
    raise RuntimeError("AGNES_API_KEY environment variable is required.")

BENCH_CLIENT = AsyncOpenAI(api_key=API_KEY, base_url=PSOAConfig.BASE_URL)
BENCH_MODEL = PSOAConfig.MODEL
TEMPERATURE = 0.3
PSOA_MAX_ROUNDS = 3


# ==================== 题库（20道，分层三类） ====================

@dataclass
class Question:
    id: int
    category: str       # 数学 | 逻辑 | 常识
    question: str
    answer: str         # 标准答案
    answer_number: Optional[float] = None


QUESTIONS: List[Question] = [
    # -- 数学计算题（6道）--
    Question(1, "数学", "一个球拍和球共花1.10美元，球拍比球贵1美元，球多少钱？",
             "0.05美元", 0.05),
    Question(2, "数学", "如果3个人3分钟能刷3面墙，那么100个人刷100面墙需要多少分钟？",
             "3分钟", 3),
    Question(3, "数学", "一个笼子里有鸡和兔共35个头，94只脚。问鸡和兔各有多少只？",
             "鸡23只，兔12只"),
    Question(4, "数学", "一张桌子的高度：站猫-蹲龟=150cm，蹲猫-站龟=110cm。求桌高。",
             "130cm", 130),
    Question(5, "数学", "甲乙相距100公里，相向而行，甲6km/h乙4km/h，狗以10km/h来回跑。相遇时狗跑了多远？",
             "100公里", 100),
    Question(6, "数学", "一个水龙头放满水池4小时，另一个6小时。一起开多久放满？",
             "2.4小时", 2.4),

    # -- 经典逻辑诡辩题（7道）--
    Question(7, "逻辑", "Monty Hall: 三扇门，你选1号，主持人打开有山羊的3号。应否换到2号？为什么？",
             "应该换，换的中奖概率2/3，不换1/3"),
    Question(8, "逻辑", "岔路口生路死路，两个守卫一真一假。问一个问题找出生路，问什么？",
             "问：如果我问另一个守卫哪条路生还，他会指哪条？然后走相反的路"),
    Question(9, "逻辑", "狼、羊、白菜过河，船每次带一样。如何安全过河？",
             "1带羊过2空回3带狼过4带羊回5带白菜过6空回7带羊过"),
    Question(10, "逻辑", "两枚硬币加起来3元，其中一枚不是1元。各是多少？",
             "一枚1元，一枚2元"),
    Question(11, "逻辑", "A说我们都是说谎者，B说至少一人说谎。已知每人要么总说真话要么总说谎。谁在说谎？",
             "A说谎，B说真话"),
    Question(12, "逻辑", "甲说乙说谎，乙说丙说谎，丙说甲乙都说谎。谁说真话？",
             "甲说谎，乙说真话，丙说谎"),
    Question(13, "逻辑", "5人戴红蓝帽子，从后往前猜自己颜色，可以说红/蓝/过。如何保证至少4人对？",
             "第5人用前4人帽子的奇偶性传递信息，后4人可推算出自己颜色"),

    # -- 常识/事实推理（7道）--
    Question(14, "常识", "所有A是B，所有B是C，则？a)所有C是A b)所有A是C c)所有C是B d)不一定",
             "b)所有A是C"),
    Question(15, "常识", "原价100元，提价20%再打八折。最终价比原价高还是低？",
             "低，最终96元"),
    Question(16, "常识", "老人生于1900年2月29日，到2020年过了多少个真正生日（2月29日）？",
             "30个"),
    Question(17, "常识", "门外3开关控制室内3盏灯，只能进一次门。如何确定对应关系？",
             "开开关1等几分钟后关，开开关2，进门。亮的是2，热的是1，冷的是3"),
    Question(18, "常识", "两人各抛硬币。两正概率？已知至少一正，两正条件概率？",
             "1/4, 1/3"),
    Question(19, "常识", "所有猫是哺乳动物，所有哺乳动物有脊椎，则必然？a)有脊椎的是猫 b)有的猫无脊椎 c)猫有脊椎 d)哺乳动物都是猫",
             "c)猫有脊椎"),
    Question(20, "常识", "细菌每分钟分裂1变2。12:00放一个，13:00满。何时半满？",
             "12:59"),
]


# ==================== A组：直接回答 ====================

async def ask_direct(question: str) -> Tuple[str, float, int, int]:
    """直接回答：单次调用，无迭代"""
    start = time.time()
    try:
        response = await BENCH_CLIENT.chat.completions.create(
            model=BENCH_MODEL,
            messages=[
                {"role": "system", "content": "你是一个助手，请直接回答问题，给出最终结论。"},
                {"role": "user", "content": question}
            ],
            temperature=TEMPERATURE,
            max_tokens=2000,
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return content, elapsed, prompt_tokens, completion_tokens
    except APIError as e:
        return f"[API Error: {e}]", time.time() - start, 0, 0
    except Exception as e:
        return f"[Error: {e}]", time.time() - start, 0, 0


# ==================== 评分机制 ====================

MATCH_CACHE = {}

def extract_number(text: str) -> Optional[float]:
    """从文本中提取第一个有意义的数字"""
    text = re.sub(r'%\s*', '', text)
    matches = re.findall(r'(\d+\.?\d*)', text.replace(',', ''))
    for m in matches:
        try:
            val = float(m)
            if 0 < val < 100000:
                return val
        except ValueError:
            continue
    return None


def numbers_close(a: float, b: float, tolerance: float = 0.01) -> bool:
    return abs(a - b) < tolerance


async def judge_with_llm(question: Question, model_output: str) -> bool:
    """LLM 评判：判断输出与标准答案是否一致"""
    cache_key = (question.id, model_output[:80])
    if cache_key in MATCH_CACHE:
        return MATCH_CACHE[cache_key]

    judge_system = """你是严格的逻辑考官。请对比【标准答案】和【模型输出】。
如果模型输出在核心逻辑、最终结论上与标准答案一致，即使表述不同，也判为正确（1），否则错误（0）。
只输出一个数字：1 或 0。不要输出其他任何内容。"""

    judge_user = f"""【标准答案】
{question.answer}

【模型输出】
{model_output}

请输出 1（正确）或 0（错误）："""

    try:
        response = await BENCH_CLIENT.chat.completions.create(
            model=BENCH_MODEL,
            messages=[
                {"role": "system", "content": judge_system},
                {"role": "user", "content": judge_user}
            ],
            temperature=0.1,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip()
        is_correct = result == "1"
        MATCH_CACHE[cache_key] = is_correct
        return is_correct
    except Exception:
        return False


async def score_answer(question: Question, model_output: str) -> bool:
    """评分：数学题数字硬匹配优先，其他用 LLM 评判"""
    if question.answer_number is not None:
        extracted = extract_number(model_output)
        if extracted is not None and numbers_close(extracted, question.answer_number):
            return True
    return await judge_with_llm(question, model_output)


# ==================== 运行结果记录 ====================

@dataclass
class SingleResult:
    question_id: int
    category: str
    direct_answer: str = ""
    direct_correct: bool = False
    direct_time: float = 0.0
    direct_tokens_in: int = 0
    direct_tokens_out: int = 0
    psoa_answer: str = ""
    psoa_correct: bool = False
    psoa_time: float = 0.0
    psoa_tokens_in: int = 0
    psoa_tokens_out: int = 0
    psoa_converged: bool = False
    psoa_rounds: int = 0


# ==================== Benchmark 主逻辑 ====================

async def run_benchmark():
    print("=" * 70)
    print("PSOA A/B Benchmark")
    print(f"Model: {BENCH_MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"PSOA max rounds: {PSOA_MAX_ROUNDS}")
    print("=" * 70)

    results: List[SingleResult] = []

    # -- 阶段一：A 组 --
    print("\n>>> Phase 1: Group A - Direct Answer <<<")
    for i, q in enumerate(QUESTIONS, 1):
        short = q.question[:40].replace('\n', ' ')
        print(f"  [{i}/{len(QUESTIONS)}] Q{q.id}[{q.category}] {short}...", end=" ")
        sys.stdout.flush()
        answer, elapsed, tok_in, tok_out = await ask_direct(q.question)
        correct = await score_answer(q, answer)
        results.append(SingleResult(
            question_id=q.id, category=q.category,
            direct_answer=answer, direct_correct=correct,
            direct_time=elapsed, direct_tokens_in=tok_in,
            direct_tokens_out=tok_out,
        ))
        mark = "OK" if correct else "NO"
        print(f"{mark}  {elapsed:.1f}s")

    # -- 阶段二：B 组 --
    print("\n>>> Phase 2: Group B - PSOA <<<")
    for i, r in enumerate(results):
        q = QUESTIONS[i]
        short = q.question[:40].replace('\n', ' ')
        print(f"  [{i+1}/{len(QUESTIONS)}] Q{q.id}[{q.category}] {short}...", end=" ")
        sys.stdout.flush()

        answer, converged, total_time, tok_in, tok_out, actual_rounds = await run_psoa_single(
            question=q.question, task_type=q.category,
            max_rounds=PSOA_MAX_ROUNDS, temperature=TEMPERATURE,
        )

        correct = await score_answer(q, answer)
        r.psoa_answer = answer
        r.psoa_correct = correct
        r.psoa_time = total_time
        r.psoa_tokens_in = tok_in
        r.psoa_tokens_out = tok_out
        r.psoa_converged = converged
        r.psoa_rounds = actual_rounds

        mark = "OK" if correct else "NO"
        conv = f"conv={actual_rounds}r" if converged else f"NC({actual_rounds}r)"
        print(f"{mark}  {total_time:.1f}s  {conv}")

    return results


# ==================== 报告生成 ====================

def generate_report(results: List[SingleResult]) -> str:
    total = len(results)
    dc = sum(1 for r in results if r.direct_correct)
    pc = sum(1 for r in results if r.psoa_correct)
    da = dc / total * 100
    pa = pc / total * 100
    dtt = sum(r.direct_time for r in results)
    ptt = sum(r.psoa_time for r in results)
    dtok = sum(r.direct_tokens_in + r.direct_tokens_out for r in results)
    ptok = sum(r.psoa_tokens_in + r.psoa_tokens_out for r in results)
    avg_r = sum(r.psoa_rounds for r in results) / total
    cc = sum(1 for r in results if r.psoa_converged)

    lines = []
    lines.append("# PSOA A/B Benchmark Report\n")
    lines.append(f"**Model:** {BENCH_MODEL} | **Temperature:** {TEMPERATURE} | **PSOA Max Rounds:** {PSOA_MAX_ROUNDS}\n")
    lines.append("## Overall\n")
    lines.append("| Metric | Direct (A) | PSOA (B) | Delta |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Accuracy** | {dc}/{total} ({da:.1f}%) | {pc}/{total} ({pa:.1f}%) | **{pa-da:+.1f}%** |")
    lines.append(f"| **Total Time** | {dtt:.1f}s | {ptt:.1f}s | +{ptt-dtt:.1f}s |")
    lines.append(f"| **Total Tokens** | {dtok:,} | {ptok:,} | +{ptok-dtok:,} |")
    lines.append(f"| **Convergence Rate** | - | {cc}/{total} ({cc/total*100:.1f}%) | - |")
    lines.append(f"| **Avg PSOA Rounds** | - | {avg_r:.2f} | - |\n")

    # 分类
    lines.append("## By Category\n")
    lines.append("| Category | Count | Direct | PSOA | Delta |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for cat in ["数学", "逻辑", "常识"]:
        cr = [r for r in results if r.category == cat]
        if not cr:
            continue
        cdc = sum(1 for r in cr if r.direct_correct)
        cpc = sum(1 for r in cr if r.psoa_correct)
        cda = cdc / len(cr) * 100
        cpa = cpc / len(cr) * 100
        lines.append(f"| {cat} | {len(cr)} | {cda:.1f}% | {cpa:.1f}% | {cpa-cda:+.1f}% |")
    lines.append("")

    lines.append("## Per Question\n")
    lines.append("| # | Cat | Direct | PSOA | D.Time | P.Time | Rounds | Conv |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        d = "OK" if r.direct_correct else "NO"
        p = "OK" if r.psoa_correct else "NO"
        c = "Y" if r.psoa_converged else "N"
        lines.append(f"| {r.question_id} | {r.category} | {d} | {p} | {r.direct_time:.1f}s | {r.psoa_time:.1f}s | {r.psoa_rounds} | {c} |")
    lines.append("")

    imp = pa - da
    ec = ptok * 0.000002
    lines.append("---")
    lines.append(f"*PSOA accuracy improvement: {imp:+.1f}%, extra time: {ptt-dtt:.0f}s, extra token cost ~${ec:.2f}, convergence: {cc/total*100:.1f}%*")
    lines.append("")

    # ---- 异常分析 ----
    false_positive = [r for r in results if r.psoa_correct == False and r.psoa_converged == True]
    false_negative = [r for r in results if r.psoa_correct == True and r.psoa_converged == False]
    regressions = [r for r in results if r.direct_correct == True and r.psoa_correct == False]
    fixes = [r for r in results if r.direct_correct == False and r.psoa_correct == True]

    lines.append("## Anomaly Analysis\n")
    lines.append("### Type A: Converged but Wrong (L false positive)\n")
    if false_positive:
        lines.append("| # | Cat | Rounds | D.Time | P.Time |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for r in false_positive:
            lines.append(f"| {r.question_id} | {r.category} | {r.psoa_rounds} | {r.direct_time:.1f}s | {r.psoa_time:.1f}s |")
    else:
        lines.append("None\n")
    lines.append("")

    lines.append("### Type B: Correct but Not Converged (R false negative)\n")
    if false_negative:
        lines.append("| # | Cat | Rounds | D.Time | P.Time |")
        lines.append("| :--- | :--- | :---: | :---: | :---: |")
        for r in false_negative:
            lines.append(f"| {r.question_id} | {r.category} | {r.psoa_rounds} | {r.direct_time:.1f}s | {r.psoa_time:.1f}s |")
    else:
        lines.append("None\n")
    lines.append("")

    lines.append("### Type C: Regression (Direct OK, PSOA NO)\n")
    if regressions:
        lines.append(f"PSOA introduced {len(regressions)} regression(s): ")
        for r in regressions:
            q = next((q for q in QUESTIONS if q.id == r.question_id), None)
            qshort = q.question[:50].replace('\n',' ') if q else ""
            lines.append(f"- **Q{r.question_id}** [{r.category}]: {qshort}")
    else:
        lines.append("None\n")
    lines.append("")

    lines.append("### Type D: Fix (Direct NO, PSOA OK)\n")
    if fixes:
        lines.append(f"PSOA fixed {len(fixes)} question(s):")
        for r in fixes:
            q = next((q for q in QUESTIONS if q.id == r.question_id), None)
            qshort = q.question[:50].replace('\n',' ') if q else ""
            lines.append(f"- **Q{r.question_id}** [{r.category}]: {qshort}")
    else:
        lines.append("None\n")
    lines.append("")

    lines.append("### Summary\n")
    lines.append(f"- **L false positive rate**: {len(false_positive)}/{sum(1 for r in results if r.psoa_converged)} converged results are wrong")
    lines.append(f"- **R false negative rate**: {len(false_negative)}/{sum(1 for r in results if r.psoa_rounds > 0)} PSOA runs failed to converge despite correct answer")
    lines.append(f"- **Net improvement**: {len(fixes)} fixes - {len(regressions)} regressions = {len(fixes)-len(regressions)} net gain")
    lines.append("")

    return "\n".join(lines)


def print_table(results):
    total = len(results)
    dc = sum(1 for r in results if r.direct_correct)
    pc = sum(1 for r in results if r.psoa_correct)
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'ID':>3} {'Cat':<5} {'Direct':>6} {'PSOA':>6} {'D.Time':>7} {'P.Time':>7} {'Rnd':>4} {'Conv':>4}")
    print("-" * 70)
    for r in results:
        d = "OK" if r.direct_correct else "NO"
        p = "OK" if r.psoa_correct else "NO"
        c = "Y" if r.psoa_converged else "N"
        print(f"{r.question_id:>3} {r.category:<5} {d:>6} {p:>6} {r.direct_time:>6.1f}s {r.psoa_time:>6.1f}s {r.psoa_rounds:>2}r {c:>4}")
    print("-" * 70)
    print(f"{'':>3} {'TOTAL':<5} {f'{dc}/{total}':>6} {f'{pc}/{total}':>6} {'':>7} {'':>7} {'':>4} {'':>4}")
    print(f"{'':>3} {'Acc':<5} {f'{dc/total*100:.1f}%':>6} {f'{pc/total*100:.1f}%':>6}")
    print("=" * 70)
    
    # 终端异常摘要
    fp = [r for r in results if not r.psoa_correct and r.psoa_converged]
    fn = [r for r in results if r.psoa_correct and not r.psoa_converged]
    reg = [r for r in results if r.direct_correct and not r.psoa_correct]
    fix = [r for r in results if not r.direct_correct and r.psoa_correct]
    if fp or fn:
        print()
        print("ANOMALIES:")
        if fp:
            print(f"  [FP] Converged but WRONG: Q{', '.join(str(r.question_id) for r in fp)}")
        if fn:
            print(f"  [FN] Correct but NO CONV: Q{', '.join(str(r.question_id) for r in fn)}")
        if reg:
            print(f"  [REG] Direct OK -> PSOA NO: Q{', '.join(str(r.question_id) for r in reg)}")
        if fix:
            print(f"  [FIX] Direct NO -> PSOA OK: Q{', '.join(str(r.question_id) for r in fix)}")


# ==================== 入口 ====================

async def main():
    results = await run_benchmark()
    print_table(results)
    report = generate_report(results)
    with open("benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: benchmark_report.md")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
