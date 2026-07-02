#!/usr/bin/env python3
"""
DeepSeek 数学推理 Benchmark（10题 A/B 测试）
=============================================
对比"直接回答" vs "PSOA架构"在 10 道数学推理题上的准确率、耗时、成本。

API: DeepSeek (从环境变量 DEEPSEEK_API_KEY 读取)
"""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from openai import AsyncOpenAI, APIError


# ==================== DeepSeek 配置 ====================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable is required.")

BENCH_CLIENT = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)
BENCH_MODEL = "deepseek-chat"
TEMPERATURE = 0.3
PSOA_MAX_ROUNDS = 3

# PSOA 内部也使用同一个 DeepSeek client
PSOA_CLIENT = BENCH_CLIENT
PSOA_MODEL = BENCH_MODEL


# ==================== 题库（10道数学推理题） ====================

@dataclass
class Question:
    id: int
    category: str       # 统一为"数学"
    question: str
    answer: str         # 标准答案（文本）
    answer_number: Optional[float] = None  # 可用于数字匹配


QUESTIONS: List[Question] = [
    Question(1, "数学",
        "一件商品原价 100 元。商家先提价 20%，再打八折（降价 20%）。请问最终价格比原价高、低，还是持平？具体是多少？",
        "低，最终 96 元", 96),
    Question(2, "数学",
        "有 5 个数的平均数是 20。如果去掉其中一个数，剩下 4 个数的平均数变为 18。请问被去掉的那个数是多少？",
        "28", 28),
    Question(3, "数学",
        "某人从山脚爬到山顶，速度是 3 公里/小时；到达山顶后立即沿原路返回，速度是 6 公里/小时。整个往返过程的平均速度是多少？",
        "4 公里/小时", 4),
    Question(4, "数学",
        "从下午 3 点整开始，时针和分针第一次完全重合是在什么时候？（精确到秒）",
        "3 点 16 分 21.8 秒"),
    Question(5, "数学",
        "同时掷两枚均匀骰子。已知至少有一枚骰子显示为 2 点，求两枚骰子点数之和为 6 的条件概率。",
        "2/11"),
    Question(6, "数学",
        "两个连续正整数的平方差是 27。求这两个数。",
        "13 和 14"),
    Question(7, "数学",
        "一个圆的半径为 r。在这个圆内画一个面积最大的正方形，这个正方形的面积是多少？",
        "2r²"),
    Question(8, "数学",
        "一本 100 页的书，第 1 页是奇数页。请问这本书中，页码为奇数的页数一共有多少页？",
        "50 页", 50),
    Question(9, "数学",
        "找规律填数字：2，6，12，20，30，？",
        "42", 42),
    Question(10, "数学",
        "一个池塘里的睡莲每天面积翻一倍。它需要 30 天覆盖整个池塘。请问睡莲覆盖池塘一半面积时，是第几天？",
        "第 29 天", 29),
]


# ==================== A组：直接回答 ====================

async def ask_direct(question: str) -> Tuple[str, float, int, int]:
    """直接回答：单次调用，无迭代"""
    start = time.time()
    try:
        response = await BENCH_CLIENT.chat.completions.create(
            model=BENCH_MODEL,
            messages=[
                {"role": "system", "content": "你是一个数学推理助手，请直接回答问题，给出最终结论和简要推理。"},
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

    judge_system = """你是严格的数学考官。请对比【标准答案】和【模型输出】。
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
    """评分：有数字答案时优先硬匹配，否则用 LLM 评判"""
    if question.answer_number is not None:
        extracted = extract_number(model_output)
        if extracted is not None and numbers_close(extracted, question.answer_number):
            return True
    return await judge_with_llm(question, model_output)


# ==================== PSOA 核心（移植自 psoa_demo_v2.py，使用 DeepSeek client） ====================

G_SYSTEM = """你是PSOA架构的【G线程·生成线程】。
你的职责：接收任务描述、当前策略提示、审查历史，生成候选输出。

核心要求：
1. 严格遵循策略提示中的方法论
2. 充分吸收审查历史中的修正建议
3. 输出完整、连贯、无自我审查的候选答案
4. 用中文回答，逻辑清晰、结构良好
5. 不要因为害怕被R抓错就过度保守——你的任务是产出完整答案"""

R_SYSTEM = """你是PSOA架构的【R线程·审查线程】，只负责纯负反馈。

你必须以极高标准审查，输出**严格的JSON格式**（不要输出JSON以外的任何内容）：

{
  "errors": [
    {
      "type": "逻辑错误|事实错误|遗漏|不一致|表述模糊|其他",
      "severity": "致命|严重|轻微",
      "description": "具体问题描述",
      "suggestion": "具体修改建议"
    }
  ],
  "overall_assessment": "一句话总结审查结论"
}

致命标准（必须严格执行，不可降级）：
- 任何事实性错误 → 致命（不管核心结论是否碰巧正确）
- 逻辑矛盾/自相矛盾 → 致命
- 关键前提遗漏导致结论无法推出 → 致命
- 只有纯表述优化、措辞建议才可标为轻微
- "严重"仅用于：推理过程有瑕疵但不影响结论成立的情况"""

L_SYSTEM = """你是PSOA架构的【L线程·放行线程】，判据是"这样说对了吗"——检验逻辑自洽性。

请严格按以下JSON格式输出（不要添加JSON以外的文字）：

{
  "internal_consistency": {"passed": true/false, "comment": "推理各步骤之间是否自洽"},
  "causal_closure": {"passed": true/false, "comment": "因果链是否完整，前提能否推出结论"},
  "semantic_completeness": {"passed": true/false, "comment": "关键概念是否定义清晰，语义是否完整"},
  "fatal_reconciliation": {"comment": "如果R线程标记了致命错误，你必须在此说明：为什么你仍然认为逻辑自洽（放行），或承认R的致命发现成立（不放行）。如果R没有标致命，填'不适用'"},
  "let_through": 1,
  "summary": "一句话最终判断理由"
}

关键规则：
- 三个维度全部通过才可 let_through = 1
- fatal_reconciliation 是必填字段：R标致命时你必须正面回应，不能无视
- 你只判断逻辑自洽性，不做质量评分
- 用中文填写comment和summary"""

I_SYSTEM = """你是PSOA架构的【I线程·代际蒸馏引擎】。
你的职责体现代际生命周期：蒸馏→繁衍→自消亡。

请严格按以下JSON格式输出：

{
  "round_experience": "本轮关键经验总结（用 bullet points，具体不空泛）",
  "strategy_updates": {
    "retained": ["本轮验证有效、继续保留的策略要点"],
    "modified": ["本轮发现需要修正的策略要点，含修正方向"],
    "new": ["本轮新发现的策略要点，来源是本轮G/R/L的交互"]
  },
  "next_strategy": "完整的下一代策略提示文本（要非常具体、可执行、比上一代更强）"
}

下一代策略必须：
- 比上一代更精准，包含本轮学到的具体技巧
- 包含明确的步骤和检查清单
- 针对任务类型持续优化
- 不要泛泛而谈，要有可执行的细节"""


async def call_llm_detailed(system: str, user: str, temperature: float = 0.7, retries: int = 2):
    """调用 DeepSeek LLM API，带重试，返回 (content, usage_dict)"""
    for attempt in range(retries + 1):
        try:
            response = await PSOA_CLIENT.chat.completions.create(
                model=PSOA_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=4000,
            )
            content = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
            }
            return content, usage
        except APIError as e:
            if attempt == retries:
                return f'[API Error: {e}]', {"prompt_tokens": 0, "completion_tokens": 0}
            await asyncio.sleep(1 * (attempt + 1))


async def call_llm(system: str, user: str, temperature: float = 0.7, retries: int = 2) -> str:
    """调用 LLM API，返回文本内容"""
    (content, _) = await call_llm_detailed(system, user, temperature, retries)
    return content


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取JSON，容错处理"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


def parse_r_result(raw: str) -> Dict[str, Any]:
    """解析R线程输出，兼容JSON和纯文本"""
    j = extract_json(raw)
    if j is not None:
        return j
    errors = []
    for line in raw.split('\n'):
        if '错误' in line or '严重' in line or '致命' in line:
            errors.append({
                "type": "未知",
                "severity": "致命" if '致命' in line else ("严重" if '严重' in line else "轻微"),
                "description": line.strip(),
                "suggestion": ""
            })
    return {
        "errors": errors,
        "overall_assessment": raw[:200] if raw else "解析失败"
    }


def parse_l_result(raw: str) -> Dict[str, Any]:
    """解析L线程输出，兼容JSON和纯文本"""
    j = extract_json(raw)
    if j is not None:
        return j
    signal = 0
    for line in raw.split('\n'):
        if '放行信号' in line:
            if '1' in line or '✓' in line:
                signal = 1
            break
    return {
        "internal_consistency": {"passed": False, "comment": "JSON解析失败"},
        "causal_closure": {"passed": False, "comment": "JSON解析失败"},
        "semantic_completeness": {"passed": False, "comment": "JSON解析失败"},
        "fatal_reconciliation": {"comment": "降级解析"},
        "let_through": signal,
        "summary": raw[:200] if raw else "解析失败"
    }


def has_fatal_error(r_parsed: Dict[str, Any]) -> bool:
    """检查R线程是否发现致命错误"""
    errors = r_parsed.get("errors", [])
    for err in errors:
        severity = err.get("severity", "")
        err_type = err.get("type", "")
        desc = err.get("description", "")
        if severity == "致命":
            return True
        if err_type in ("事实错误", "逻辑错误") and severity in ("严重", "致命"):
            return True
        if any(kw in desc for kw in ["事实错误", "逻辑矛盾", "自相矛盾"]):
            return True
    overall = r_parsed.get("overall_assessment", "")
    if '致命' in overall:
        return True
    return False


def candidate_similarity(a: str, b: str) -> float:
    """计算两个候选答案的token级Jaccard相似度"""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


REVIEW_WINDOW = 3
REVIEW_SUMMARY_MAX = 800


def compress_review_history(history_entries: list[str], window: int = REVIEW_WINDOW) -> str:
    """滑动窗口 + 压缩"""
    if len(history_entries) <= window:
        return "\n".join(history_entries)
    recent = history_entries[-window:]
    older = history_entries[:-window]
    compressed = f"[历史摘要·前{len(older)}轮主要问题] "
    key_issues = []
    for entry in older:
        if '致命' in entry:
            key_issues.append("含致命错误")
        elif '严重' in entry:
            key_issues.append("含严重问题")
        else:
            key_issues.append("轻微问题")
    compressed += " → ".join(key_issues)
    return compressed + "\n\n" + "\n".join(recent)


async def run_psoa_single(question: str, task_type: str = "数学", max_rounds: int = 3, temperature: float = 0.3):
    """
    单次 PSOA 运行（使用 DeepSeek API）。
    返回: (final_answer, converged, total_time, total_input_tokens, total_output_tokens, actual_rounds)
    """
    strategy = f"""【第0代初始策略】
任务类型: {task_type}
方法论:
1. 仔细分析问题，识别核心前提与结论
2. 逐步推理，确保每一步都有充分依据
3. 验证结论是否由前提逻辑推出
4. 检查是否有遗漏的中间环节
5. 用清晰的语言组织答案"""

    review_entries = []
    round_signals = []
    final_answer = ""
    converged = False
    last_passing_answer = ""
    total_input_tokens = 0
    total_output_tokens = 0

    start_time = time.time()

    for round_num in range(1, max_rounds + 1):
        # --- G 线程 ---
        g_user = f"""任务：{question}

当前策略提示：
{strategy}

审查历史（请充分吸收修正建议）：
{compress_review_history(review_entries) if review_entries else '（首轮，暂无审查历史）'}

请根据策略提示生成候选输出。"""
        candidate_raw = await call_llm_detailed(G_SYSTEM, g_user, temperature=temperature)
        candidate = candidate_raw[0]
        g_usage = candidate_raw[1]
        total_input_tokens += g_usage.get("prompt_tokens", 0)
        total_output_tokens += g_usage.get("completion_tokens", 0)

        # --- R 线程 ---
        r_user = f"""请对以下候选输出进行纯负反馈审查：

---
{candidate}
---

请输出严格的JSON格式审查结果。"""
        r_raw = await call_llm_detailed(R_SYSTEM, r_user, temperature=temperature)
        r_text = r_raw[0]
        r_usage = r_raw[1]
        total_input_tokens += r_usage.get("prompt_tokens", 0)
        total_output_tokens += r_usage.get("completion_tokens", 0)
        r_parsed = parse_r_result(r_text)

        # --- L 线程（传入R的致命信息） ---
        fatal_errors = []
        for err in r_parsed.get("errors", []):
            if err.get("severity") in ("致命",):
                fatal_errors.append(f"- [{err.get('type')}] {err.get('description')}")
        r_fatal_info = ""
        if fatal_errors:
            r_fatal_info = f"""

⚠️ R线程标记了以下致命错误（你必须在fatal_reconciliation字段中正面回应）：
{chr(10).join(fatal_errors)}
如果你认为R的致命判断不成立，请在fatal_reconciliation中给出理由；如果成立，let_through必须为0。"""
        l_user = f"""原始问题：{question}

候选输出：
{candidate}{r_fatal_info}

请检验此候选输出的逻辑自洽性，输出严格JSON格式。"""
        l_raw = await call_llm_detailed(L_SYSTEM, l_user, temperature=temperature)
        l_text = l_raw[0]
        l_usage = l_raw[1]
        total_input_tokens += l_usage.get("prompt_tokens", 0)
        total_output_tokens += l_usage.get("completion_tokens", 0)
        l_parsed = parse_l_result(l_text)

        signal = l_parsed.get("let_through", 0)
        fatal = has_fatal_error(r_parsed)

        # --- I 线程 ---
        i_user = f"""任务：{question}

当前代（第{round_num}代）策略提示：
{strategy}

本轮G线程候选输出：
{candidate}

本轮R线程审查意见：
{r_text}

本轮L线程放行判断：
{l_text}

轮次：{round_num}

请执行代际蒸馏，输出严格JSON格式。"""
        i_raw = await call_llm_detailed(I_SYSTEM, i_user, temperature=temperature)
        i_text = i_raw[0]
        i_usage = i_raw[1]
        total_input_tokens += i_usage.get("prompt_tokens", 0)
        total_output_tokens += i_usage.get("completion_tokens", 0)
        i_parsed = extract_json(i_text)
        if i_parsed is None:
            i_parsed = {"round_experience": i_text[:200], "strategy_updates": {"retained": [], "modified": [], "new": []}, "next_strategy": strategy}

        # 更新策略
        new_strategy = i_parsed.get("next_strategy", strategy)
        if len(new_strategy) < 50:
            new_strategy = strategy

        r_entry = f"第{round_num}轮: " + json.dumps(r_parsed.get("errors", []), ensure_ascii=False)[:REVIEW_SUMMARY_MAX]
        review_entries.append(r_entry)
        round_signals.append(signal)

        # 收敛判断
        if signal == 1 and not fatal:
            final_answer = candidate
            converged = True
            break
        else:
            strategy = new_strategy

    total_time = time.time() - start_time
    actual_rounds = len(round_signals)

    if not converged and round_signals:
        final_answer = candidate

    return final_answer, converged, total_time, total_input_tokens, total_output_tokens, actual_rounds


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
    print("DeepSeek 数学推理 Benchmark（10题）")
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
    lines.append("# DeepSeek 数学推理 Benchmark 报告\n")
    lines.append(f"**Model:** {BENCH_MODEL} | **Temperature:** {TEMPERATURE} | **PSOA Max Rounds:** {PSOA_MAX_ROUNDS}\n")
    lines.append("## Overall\n")
    lines.append("| Metric | Direct (A) | PSOA (B) | Delta |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Accuracy** | {dc}/{total} ({da:.1f}%) | {pc}/{total} ({pa:.1f}%) | **{pa-da:+.1f}%** |")
    lines.append(f"| **Total Time** | {dtt:.1f}s | {ptt:.1f}s | +{ptt-dtt:.1f}s |")
    lines.append(f"| **Total Tokens** | {dtok:,} | {ptok:,} | +{ptok-dtok:,} |")
    lines.append(f"| **Convergence Rate** | - | {cc}/{total} ({cc/total*100:.1f}%) | - |")
    lines.append(f"| **Avg PSOA Rounds** | - | {avg_r:.2f} | - |\n")

    lines.append("## Per Question\n")
    lines.append("| # | Direct | PSOA | D.Time | P.Time | Rounds | Conv |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        d = "OK" if r.direct_correct else "NO"
        p = "OK" if r.psoa_correct else "NO"
        c = "Y" if r.psoa_converged else "N"
        lines.append(f"| {r.question_id} | {d} | {p} | {r.direct_time:.1f}s | {r.psoa_time:.1f}s | {r.psoa_rounds} | {c} |")
    lines.append("")

    imp = pa - da
    ec = ptok * 0.000002
    lines.append("---")
    lines.append(f"*PSOA accuracy improvement: {imp:+.1f}%, extra time: {ptt-dtt:.0f}s, extra token cost ~${ec:.2f}*")
    lines.append("")

    # 异常分析
    regressions = [r for r in results if r.direct_correct and not r.psoa_correct]
    fixes = [r for r in results if not r.direct_correct and r.psoa_correct]

    lines.append("## Analysis\n")
    lines.append(f"**Regressions** (Direct OK → PSOA NO): {len(regressions)}")
    if regressions:
        for r in regressions:
            q = next((q for q in QUESTIONS if q.id == r.question_id), None)
            qshort = q.question[:50].replace('\n',' ') if q else ""
            lines.append(f"- Q{r.question_id}: {qshort}")
    lines.append("")
    lines.append(f"**Fixes** (Direct NO → PSOA OK): {len(fixes)}")
    if fixes:
        for r in fixes:
            q = next((q for q in QUESTIONS if q.id == r.question_id), None)
            qshort = q.question[:50].replace('\n',' ') if q else ""
            lines.append(f"- Q{r.question_id}: {qshort}")
    lines.append("")
    lines.append(f"**Net improvement**: {len(fixes)} fixes - {len(regressions)} regressions = {len(fixes)-len(regressions)} net gain")
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
    print(f"{'ID':>3} {'Direct':>6} {'PSOA':>6} {'D.Time':>7} {'P.Time':>7} {'Rnd':>4} {'Conv':>4}")
    print("-" * 55)
    for r in results:
        d = "OK" if r.direct_correct else "NO"
        p = "OK" if r.psoa_correct else "NO"
        c = "Y" if r.psoa_converged else "N"
        print(f"{r.question_id:>3} {d:>6} {p:>6} {r.direct_time:>6.1f}s {r.psoa_time:>6.1f}s {r.psoa_rounds:>2}r {c:>4}")
    print("-" * 55)
    print(f"{'':>3} {'TOTAL':>6} {f'{dc}/{total}':>6} {'':>7} {'':>7} {'':>4} {'':>4}")
    print(f"{'':>3} {'Acc':>6} {f'{dc/total*100:.1f}%':>6}")
    print("=" * 70)


# ==================== 入口 ====================

async def main():
    results = await run_benchmark()
    print_table(results)
    report = generate_report(results)
    report_path = "benchmark_deepseek_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
