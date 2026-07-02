#!/usr/bin/env python3
"""PSOA 10题快速测试 - 从20题题库中抽取10道（3数学+4逻辑+3常识）"""
import asyncio
import json
import time
import sys
import os

sys.path.insert(0, str(os.path.dirname(__file__)))
from psoa_demo_v2 import Config, run_psoa_single
from openai import AsyncOpenAI, APIError

BENCH_CLIENT = AsyncOpenAI(api_key=Config.API_KEY(), base_url=Config.BASE_URL)
BENCH_MODEL = Config.MODEL
TEMPERATURE = 0.3
PSOA_MAX_ROUNDS = 3

# 10道题：从benchmark.py原20题中抽取
QUESTIONS = [
    {"id": 1, "cat": "数学", "q": "一个球拍和球共花1.10美元，球拍比球贵1美元，球多少钱？", "a": "0.05美元", "num": 0.05},
    {"id": 4, "cat": "数学", "q": "一张桌子的高度：站猫-蹲龟=150cm，蹲猫-站龟=110cm。求桌高。", "a": "130cm", "num": 130},
    {"id": 6, "cat": "数学", "q": "一个水龙头放满水池4小时，另一个6小时。一起开多久放满？", "a": "2.4小时", "num": 2.4},
    {"id": 7, "cat": "逻辑", "q": "Monty Hall: 三扇门，你选1号，主持人打开有山羊的3号。应否换到2号？为什么？", "a": "应该换，换的中奖概率2/3，不换1/3"},
    {"id": 9, "cat": "逻辑", "q": "狼、羊、白菜过河，船每次带一样。如何安全过河？", "a": "1带羊过2空回3带狼过4带羊回5带白菜过6空回7带羊过"},
    {"id": 12, "cat": "逻辑", "q": "甲说乙说谎，乙说丙说谎，丙说甲乙都说谎。谁说真话？", "a": "甲说谎，乙说真话，丙说谎"},
    {"id": 13, "cat": "逻辑", "q": "5人戴红蓝帽子，从后往前猜自己颜色，可以说红/蓝/过。如何保证至少4人对？", "a": "第5人用前4人帽子的奇偶性传递信息，后4人可推算出自己颜色"},
    {"id": 14, "cat": "常识", "q": "所有A是B，所有B是C，则？a)所有C是A b)所有A是C c)所有C是B d)不一定", "a": "b)所有A是C"},
    {"id": 16, "cat": "常识", "q": "老人生于1900年2月29日，到2020年过了多少个真正生日（2月29日）？", "a": "30个"},
    {"id": 20, "cat": "常识", "q": "细菌每分钟分裂1变2。12:00放一个，13:00满。何时半满？", "a": "12:59"},
]

async def ask_direct(question: str):
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
        return content, elapsed
    except Exception as e:
        return f"[Error: {e}]", time.time() - start

async def judge_with_llm(q_text, std_answer, model_output):
    judge_system = """你是严格的逻辑考官。请对比【标准答案】和【模型输出】。
如果模型输出在核心逻辑、最终结论上与标准答案一致，即使表述不同，也判为正确（1），否则错误（0）。
只输出一个数字：1 或 0。不要输出其他任何内容。"""
    judge_user = f"""【标准答案】\n{std_answer}\n\n【模型输出】\n{model_output}\n\n请输出 1（正确）或 0（错误）："""
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
        return result == "1"
    except Exception:
        return False

import re
def extract_number(text):
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

async def score_answer(q, model_output):
    if q.get("num") is not None:
        extracted = extract_number(model_output)
        if extracted is not None and abs(extracted - q["num"]) < 0.01:
            return True
    return await judge_with_llm(q["q"], q["a"], model_output)

async def main():
    print("=" * 60)
    print("PSOA 10题快速测试")
    print(f"模型: {BENCH_MODEL} | 温度: {TEMPERATURE} | PSOA最大轮数: {PSOA_MAX_ROUNDS}")
    print("=" * 60)

    results = []
    total_direct_time = 0
    total_psoa_time = 0

    for i, q in enumerate(QUESTIONS):
        print(f"\n{'='*50}")
        print(f"Q{q['id']} [{q['cat']}] {q['q'][:50]}...")
        print(f"{'='*50}")

        # A: 直接回答
        print(f"  [A] 直接回答中...")
        direct_answer, d_time = await ask_direct(q["q"])
        direct_correct = await score_answer(q, direct_answer)
        total_direct_time += d_time
        d_mark = "✓" if direct_correct else "✗"
        print(f"  [A] {d_mark} ({d_time:.1f}s) {direct_answer[:80]}...")

        # B: PSOA
        print(f"  [B] PSOA运行中...")
        psoa_answer, converged, p_time, _, _, rounds = await run_psoa_single(
            question=q["q"],
            task_type=q["cat"],
            max_rounds=PSOA_MAX_ROUNDS,
            temperature=TEMPERATURE,
        )
        psoa_correct = await score_answer(q, psoa_answer)
        total_psoa_time += p_time
        p_mark = "✓" if psoa_correct else "✗"
        conv_mark = "收敛" if converged else "未收敛"
        print(f"  [B] {p_mark} ({p_time:.1f}s, {rounds}轮, {conv_mark}) {psoa_answer[:80]}...")

        results.append({
            "id": q["id"],
            "cat": q["cat"],
            "direct_correct": direct_correct,
            "psoa_correct": psoa_correct,
            "d_time": round(d_time, 1),
            "p_time": round(p_time, 1),
            "rounds": rounds,
            "converged": converged,
        })

    # 汇总
    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")

    d_correct = sum(1 for r in results if r["direct_correct"])
    p_correct = sum(1 for r in results if r["psoa_correct"])
    conv_count = sum(1 for r in results if r["converged"])

    print(f"\n| 指标 | 直接回答 | PSOA | 变化 |")
    print(f"|---|---|---|---|")
    print(f"| 准确率 | {d_correct}/10 ({d_correct*10}%) | {p_correct}/10 ({p_correct*10}%) | {(p_correct-d_correct)*10:+d}% |")
    print(f"| 总耗时 | {total_direct_time:.1f}s | {total_psoa_time:.1f}s | +{total_psoa_time-total_direct_time:.1f}s |")
    print(f"| 收敛率 | - | {conv_count}/10 ({conv_count*10}%) | - |")

    avg_rounds = sum(r["rounds"] for r in results) / len(results)
    print(f"| 平均轮数 | - | {avg_rounds:.2f} | - |")

    print(f"\n| # | 类别 | 直接 | PSOA | D.耗时 | P.耗时 | 轮数 | 收敛 |")
    print(f"|---|---|---|---|---|---|---|---|")
    for r in results:
        d_m = "✓" if r["direct_correct"] else "✗"
        p_m = "✓" if r["psoa_correct"] else "✗"
        c_m = "Y" if r["converged"] else "N"
        print(f"| Q{r['id']} | {r['cat']} | {d_m} | {p_m} | {r['d_time']}s | {r['p_time']}s | {r['rounds']} | {c_m} |")

    # 退步和修复分析
    fixes = [r for r in results if not r["direct_correct"] and r["psoa_correct"]]
    regressions = [r for r in results if r["direct_correct"] and not r["psoa_correct"]]
    print(f"\n修复: {len(fixes)}题 {[f'Q{f[\"id\"]}' for f in fixes]}")
    print(f"退步: {len(regressions)}题 {[f'Q{r[\"id\"]}' for r in regressions]}")
    print(f"净收益: {len(fixes) - len(regressions)}题")

if __name__ == "__main__":
    asyncio.run(main())
