#!/usr/bin/env python3
"""
PSOA v2 - 结构化升级版
基于 v1 原型 + ChatGPT/Grok 评审意见改进：
  1. R/L/I 三线程全部 JSON 结构化输出 + extract_json 容错
  2. I 线程策略进化使用 retained/modified/new 结构
  3. review_history 改为滑动窗口 + 压缩摘要，不再全文累积
  4. L 线程新增维度：R 标致命时必须说明为何仍放行（R-L 对冲显式化）
  5. 保留中文过程日志（v1 核心优点）
  6. Config 类整理参数
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from openai import AsyncOpenAI, APIError

# ==================== 配置区 ====================
class Config:
    RESULT_MODE = sys.argv[1] if len(sys.argv) > 1 else "notify"
    QUESTION = sys.argv[2] if len(sys.argv) > 2 else "如果所有的猫都是动物，所有的动物都是生物，那么所有的猫都是生物吗？请详细解释你的推理过程。"
    MAX_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    TASK_TYPE = sys.argv[4] if len(sys.argv) > 4 else "推理"
    
    OUTPUT_DIR = Path("/app/data/所有对话/主对话/PSOA/output")
    _API_KEY_RAW: str = os.environ.get("AGNES_API_KEY")

    @classmethod
    def API_KEY(cls) -> str:
        if not cls._API_KEY_RAW:
            raise RuntimeError("AGNES_API_KEY environment variable is required. Set it before running.")
        return cls._API_KEY_RAW
    BASE_URL = "https://apihub.agnes-ai.com/v1"
    MODEL = "agnes-2.0-flash"
    
    # 滑动窗口：最多保留最近N轮审查历史
    REVIEW_WINDOW = 3
    # 审查历史压缩摘要长度上限
    REVIEW_SUMMARY_MAX = 800


try:
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    Config.OUTPUT_DIR = Path("psoa_output")
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
client = AsyncOpenAI(api_key=Config.API_KEY(), base_url=Config.BASE_URL)


# ==================== 升级后的 System Prompts ====================

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


# ==================== 工具函数 ====================

async def call_llm(system: str, user: str, temperature: float = 0.7, retries: int = 2) -> str:
    """调用 LLM API，带重试，返回文本内容"""
    (content, _) = await call_llm_detailed(system, user, temperature, retries)
    return content


async def call_llm_detailed(system: str, user: str, temperature: float = 0.7, retries: int = 2):
    """调用 LLM API，带重试，返回 (content, usage_dict)"""
    for attempt in range(retries + 1):
        try:
            response = await client.chat.completions.create(
                model=Config.MODEL,
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


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取JSON，容错处理。解析失败返回None并在stderr打印警告。"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    print(f"[WARN] extract_json: Failed to parse JSON from LLM output (length={len(text)}), returning None")
    return None


def parse_r_result(raw: str) -> Dict[str, Any]:
    """解析R线程输出，兼容JSON和纯文本"""
    j = extract_json(raw)
    if j is not None:
        return j
    # 降级：纯文本解析
    errors = []
    has_fatal = '致命' in raw
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
    # 降级：从纯文本中提取放行信号
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
    """检查R线程是否发现致命错误（代码层+JSON层双重判定）"""
    errors = r_parsed.get("errors", [])
    for err in errors:
        severity = err.get("severity", "")
        err_type = err.get("type", "")
        desc = err.get("description", "")
        # 1. R自己标了致命
        if severity == "致命":
            return True
        # 2. 代码层强制升级：事实错误/逻辑矛盾一律致命
        if err_type in ("事实错误", "逻辑错误") and severity in ("严重", "致命"):
            return True
        if any(kw in desc for kw in ["事实错误", "逻辑矛盾", "自相矛盾"]):
            return True
    # 3. 纯文本降级时的检查
    overall = r_parsed.get("overall_assessment", "")
    if '致命' in overall:
        return True
    return False


def candidate_similarity(a: str, b: str) -> float:
    """计算两个候选答案的token级Jaccard相似度，用于收敛稳定性判断。"""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compress_review_history(history_entries: list[str], window: int = Config.REVIEW_WINDOW) -> str:
    """滑动窗口 + 压缩：只保留最近N轮审查历史，超出部分压缩为一行摘要"""
    if len(history_entries) <= window:
        return "\n".join(history_entries)
    
    # 保留最近N轮完整内容
    recent = history_entries[-window:]
    # 更早的轮次压缩为一行摘要
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


# ==================== 线程函数 ====================

async def g_thread(question: str, strategy: str, review_history: str) -> str:
    """G线程：生成候选输出"""
    user_prompt = f"""任务：{question}

当前策略提示：
{strategy}

审查历史（请充分吸收修正建议）：
{review_history if review_history else '（首轮，暂无审查历史）'}

请根据策略提示生成候选输出。"""
    return await call_llm(G_SYSTEM, user_prompt, temperature=0.75)


async def r_thread(candidate: str) -> str:
    """R线程：纯负反馈审查"""
    user_prompt = f"""请对以下候选输出进行纯负反馈审查：

---
{candidate}
---

请输出严格的JSON格式审查结果。"""
    return await call_llm(R_SYSTEM, user_prompt, temperature=0.3)


async def l_thread(candidate: str, question: str, r_parsed: Dict[str, Any]) -> str:
    """L线程：逻辑自洽性放行检验"""
    # 构建R线程致命信息，让L正面回应
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

    user_prompt = f"""原始问题：{question}

候选输出：
{candidate}
{r_fatal_info}

请检验此候选输出的逻辑自洽性，输出严格JSON格式。"""
    return await call_llm(L_SYSTEM, user_prompt, temperature=0.2)


async def l_thread_raw(candidate: str, question: str, r_raw: str) -> str:
    """L线程原始版：接收R的原始文本，不做预解析，以便与R真正并发。"""
    fatal_errors = []
    for ln in r_raw.split("\n"):
        if "致命" in ln:
            fatal_errors.append(ln.strip())
    
    r_fatal_info = ""
    if fatal_errors:
        r_fatal_info = "\n\n\u26a0\ufe0f R线程标记了致命错误（你必须在fatal_reconciliation字段中正面回应）：\n"
        for fe in fatal_errors[:3]:
            r_fatal_info += f"- {fe}\n"
        r_fatal_info += "如果你认为R的致命判断不成立，请在fatal_reconciliation中给出理由；如果成立，let_through必须为0。"
    
    user_prompt = f"""原始问题：{question}

候选输出：
{candidate}
{r_fatal_info}

请检验此候选输出的逻辑自洽性，输出严格JSON格式。"""
    return await call_llm(L_SYSTEM, user_prompt, temperature=0.2)


async def i_thread(question: str, strategy: str, candidate: str, 
                   r_summary: str, l_summary: str, round_num: int) -> Dict:
    """I线程：代际蒸馏引擎"""
    user_prompt = f"""任务：{question}

当前代（第{round_num}代）策略提示：
{strategy}

本轮G线程候选输出（摘要）：
{candidate[:800]}

本轮R线程审查摘要：
{r_summary}

本轮L线程放行判断摘要：
{l_summary}

轮次：{round_num}

请执行代际蒸馏，输出严格JSON格式。"""
    result = await call_llm(I_SYSTEM, user_prompt, temperature=0.5)
    parsed = extract_json(result)
    if parsed is None:
        parsed = {"round_experience": result[:200], "strategy_updates": {"retained": [], "modified": [], "new": []}, "next_strategy": strategy}
    return parsed


# ==================== 过程报告 ====================

def format_round_report(round_num: int, max_r: int, strategy: str, candidate: str,
                        r_parsed: Dict, l_parsed: Dict, i_parsed: Dict,
                        signal: int, fatal: bool, elapsed: float) -> str:
    """格式化单轮中文过程报告（保留v1核心优点）"""
    def trunc(text, limit=500):
        if isinstance(text, dict) or isinstance(text, list):
            text = json.dumps(text, ensure_ascii=False, indent=2)
        if len(str(text)) > limit:
            return str(text)[:limit] + f"...（共{len(str(text))}字）"
        return str(text)

    signal_display = "✓ 放行" if signal == 1 else "✗ 不放行"
    fatal_display = "⚠️ 有致命错误" if fatal else "无致命错误"

    # R线程结构化展示
    r_errors = r_parsed.get("errors", [])
    r_summary = r_parsed.get("overall_assessment", "无")
    r_display = f"  审查总结: {r_summary}\n"
    for i, err in enumerate(r_errors[:5], 1):
        r_display += f"  错误{i}: [{err.get('severity','?')}] {err.get('type','?')} - {trunc(err.get('description',''), 150)}\n"
        if err.get('suggestion'):
            r_display += f"         建议: {trunc(err['suggestion'], 100)}\n"

    # L线程结构化展示
    ic = l_parsed.get("internal_consistency", {})
    cc = l_parsed.get("causal_closure", {})
    sc = l_parsed.get("semantic_completeness", {})
    fr_raw = l_parsed.get("fatal_reconciliation", {}); fr = fr_raw.get("comment", "不适用") if isinstance(fr_raw, dict) else str(fr_raw)
    l_display = f"  内部一致性: {'✓' if ic.get('passed') else '✗'} {ic.get('comment', '')}\n"
    l_display += f"  因果闭合性: {'✓' if cc.get('passed') else '✗'} {cc.get('comment', '')}\n"
    l_display += f"  语义完整性: {'✓' if sc.get('passed') else '✗'} {sc.get('comment', '')}\n"
    l_display += f"  致命对冲: {fr}\n"
    l_display += f"  判断理由: {l_parsed.get('summary', '')}\n"

    # I线程结构化展示
    su = i_parsed.get("strategy_updates", {})
    i_display = f"  本轮经验: {trunc(i_parsed.get('round_experience', ''), 300)}\n"
    i_display += f"  保留策略: {su.get('retained', [])}\n"
    i_display += f"  修正策略: {su.get('modified', [])}\n"
    i_display += f"  新增策略: {su.get('new', [])}\n"

    report = f"""
{'='*40}
第 {round_num} 轮 / 最多 {max_r} 轮  |  耗时 {elapsed:.1f}s
{'='*40}

【G线程·生成】
  策略提示: {trunc(strategy, 300)}
  候选输出: {trunc(candidate)}

【R线程·审查】  {fatal_display}
{r_display}

【L线程·放行】  {signal_display}
{l_display}

【I线程·代际蒸馏】
{i_display}

{'='*40}
"""
    return report


# ==================== 核心单次运行函数（供 Benchmark 调用） ====================

async def run_psoa_single(question: str, task_type: str = "推理", max_rounds: int = 3, temperature: float = 0.3):
    """
    干净的单次 PSOA 运行，无日志/无提交。
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
        # G 线程
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

        # R 线程
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

        # L 线程（传入R的致命信息）
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

        # I 线程
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

        r_entry = f"第{round_num}轮: " + json.dumps(r_parsed.get("errors", []), ensure_ascii=False)[:Config.REVIEW_SUMMARY_MAX]
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
        # 取最后一轮输出作为答案
        final_answer = candidate

    return final_answer, converged, total_time, total_input_tokens, total_output_tokens, actual_rounds


# ==================== 主逻辑 ====================

async def main() -> None:
    print(f"[PSOA v2] result_mode={Config.RESULT_MODE}, question={Config.QUESTION[:50]}..., max_rounds={Config.MAX_ROUNDS}, task_type={Config.TASK_TYPE}")

    # 初始策略
    strategy = f"""【第0代初始策略】
任务类型: {Config.TASK_TYPE}
方法论:
1. 仔细分析问题，识别核心前提与结论
2. 逐步推理，确保每一步都有充分依据
3. 验证结论是否由前提逻辑推出
4. 检查是否有遗漏的中间环节
5. 用清晰的语言组织答案"""

    review_entries = []  # 滑动窗口式审查历史
    all_reports = []
    round_signals = []
    experience_log = []
    final_answer = ""
    converged = False
    last_passing_answer = ""

    start_time = time.time()

    for round_num in range(1, Config.MAX_ROUNDS + 1):
        round_start = time.time()
        print(f"\n>>> 第 {round_num} 轮开始...")

        # G线程
        print(f"  [G] 生成中...")
        candidate = await g_thread(Config.QUESTION, strategy, compress_review_history(review_entries))
        print(f"  [G] 完成, len={len(candidate)}")

        # R线程：审查候选输出
        print(f"  [R] 审查中...")
        r_raw = await r_thread(candidate)
        r_parsed = parse_r_result(r_raw)
        print(f"  [R] 完成, errors={len(r_parsed.get('errors', []))}")

        # L线程：传入R的原始文本（不做Python端JSON预解析），减少依赖
        print(f"  [L] 放行判断中...")
        l_raw = await l_thread_raw(candidate, Config.QUESTION, r_raw)
        l_parsed = parse_l_result(l_raw)
        print(f"  [L] 完成, let_through={l_parsed.get('let_through', 0)}")

        # 解析信号
        signal = l_parsed.get("let_through", 0)
        fatal = has_fatal_error(r_parsed)
        round_signals.append(signal)
        print(f"  [判断] 放行={signal}, 致命={fatal}")

        # I线程
        print(f"  [I] 代际蒸馏中...")
        # I线程：传入结构化摘要而非原始文本，节省token
        r_summary = json.dumps(r_parsed.get("errors", []), ensure_ascii=False, indent=2)[:600]
        l_summary = json.dumps({k: v for k, v in l_parsed.items() if k != "fatal_reconciliation"}, ensure_ascii=False, indent=2)[:400]
        fr = l_parsed.get("fatal_reconciliation", {})
        if isinstance(fr, dict):
            l_summary += f"\n致命对冲: {fr.get('comment', '')}"
        else:
            l_summary += f"\n致命对冲: {fr}"
        i_parsed = await i_thread(Config.QUESTION, strategy, candidate, r_summary, l_summary, round_num)
        print(f"  [I] 完成")

        # 更新策略
        new_strategy = i_parsed.get("next_strategy", strategy)
        if len(new_strategy) < 50:
            new_strategy = strategy  # I输出异常时保持原策略

        # 更新审查历史（滑动窗口）
        r_entry = f"第{round_num}轮: " + json.dumps(r_parsed.get("errors", []), ensure_ascii=False)[:Config.REVIEW_SUMMARY_MAX]
        review_entries.append(r_entry)

        # 记录经验
        experience_log.append({
            "round": round_num,
            "signal": signal,
            "fatal": fatal,
            "errors_count": len(r_parsed.get("errors", [])),
            "fatal_reconciliation": (lambda v: v.get("comment", "") if isinstance(v, dict) else str(v))(l_parsed.get("fatal_reconciliation", "")),
            "strategy_updates": i_parsed.get("strategy_updates", {}),
        })

        # 格式化报告
        elapsed = time.time() - round_start
        report = format_round_report(
            round_num, Config.MAX_ROUNDS, strategy, candidate,
            r_parsed, l_parsed, i_parsed, signal, fatal, elapsed
        )
        all_reports.append(report)
        print(report)

        # 收敛判断：L放行 + 无致命错误 + 答案稳定性（与上一轮放行答案相似度>0.85）
        if signal == 1 and not fatal:
            if converged is False:  # First time we see a passing signal
                final_answer = candidate
                last_passing_answer = candidate
                converged = True
                print(f">>> ✓ 第 {round_num} 轮首次放行，记录基准答案。")
            else:
                # Check stability against last passing answer
                sim = candidate_similarity(last_passing_answer, candidate)
                if sim > 0.85:
                    final_answer = candidate
                    print(f">>> ✓ 第 {round_num} 轮收敛！L放行 + 无致命错误 + 答案稳定(相似度={sim:.2f})。")
                    break
                else:
                    print(f">>> ⚠ 第 {round_num} 轮放行但答案变化较大(相似度={sim:.2f})，继续进化...")
                    last_passing_answer = candidate
                    strategy = new_strategy
        else:
            strategy = new_strategy
            reason = "R标致命" if fatal else "L不放行"
            print(f">>> ✗ 第 {round_num} 轮未收敛（{reason}），进入下一代...")

    total_time = time.time() - start_time

    # 最终摘要
    signal_display = " → ".join(["✓" if s == 1 else "✗" for s in round_signals])
    final_summary = f"""
{'='*40}
PSOA v2 最终摘要
{'='*40}

问题: {Config.QUESTION}
任务类型: {Config.TASK_TYPE}
总轮数: {len(round_signals)} / {Config.MAX_ROUNDS}
是否收敛: {'是 ✓' if converged else '否 ✗'}
总耗时: {total_time:.1f}s

各轮信号: {signal_display}

逐轮详情:
"""
    for exp in experience_log:
        sig = "✓" if exp["signal"] == 1 else "✗"
        fatal_mark = " [致命]" if exp["fatal"] else ""
        fr = exp.get("fatal_reconciliation", "")
        fr_display = f" | 对冲: {fr[:80]}" if fr else ""
        su = exp.get("strategy_updates", {})
        su_display = f" 保留{len(su.get('retained',[]))} 修正{len(su.get('modified',[]))} 新增{len(su.get('new',[]))}"
        final_summary += f"  第{exp['round']}轮: 放行={sig}{fatal_mark} 错误数={exp['errors_count']}{fr_display}{su_display}\n"

    if final_answer:
        final_summary += f"\n最终答案:\n{final_answer}\n"

    print(final_summary)

    # 保存日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"psoa_v2_log_{timestamp}.md"
    log_path = Config.OUTPUT_DIR / log_filename

    full_log = f"""# PSOA v2 并行自优化架构 - 完整过程日志

## 基本信息
- 版本: v2（结构化升级版）
- 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 问题: {Config.QUESTION}
- 任务类型: {Config.TASK_TYPE}
- 最大轮数: {Config.MAX_ROUNDS}
- 实际轮数: {len(round_signals)}
- 是否收敛: {'是' if converged else '否'}
- 总耗时: {total_time:.1f}s

## v2 改进点
1. R/L/I 三线程 JSON 结构化输出 + extract_json 容错
2. I 线程策略进化 retained/modified/new 结构
3. review_history 滑动窗口 + 压缩摘要
4. L 线程 fatal_reconciliation：R标致命时必须正面回应
5. 保留中文过程日志

## 迭代过程

{''.join(all_reports)}

## 最终摘要
{final_summary}
"""

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(full_log)
    print(f"\n[日志] 已保存: {log_path}")

    # 提交结果
    msg_parts = [
        f"PSOA v2 原型验证完成",
        f"问题: {Config.QUESTION[:60]}{'...' if len(Config.QUESTION) > 60 else ''}",
        f"收敛: {'是' if converged else '否'} | 轮数: {len(round_signals)}/{Config.MAX_ROUNDS} | 耗时: {total_time:.1f}s",
        f"信号: {signal_display}",
    ]
    if converged and final_answer:
        msg_parts.append(f"答案摘要: {final_answer[:200]}")
    else:
        msg_parts.append("未收敛。")

    try:
        from codeact_sdk import CodeActSDK
        sdk = CodeActSDK()
        await sdk.submit_result(
            result_mode=Config.RESULT_MODE if Config.RESULT_MODE != "auto" else "notify",
            status="success",
            message="\n".join(msg_parts),
            data={
                "converged": converged,
                "rounds": len(round_signals),
                "max_rounds": Config.MAX_ROUNDS,
                "signals": round_signals,
                "log_file": str(log_path),
                "final_answer": final_answer[:500] if final_answer else "",
            },
        )
    except Exception as e:
        print(f"[提交失败] {e}")
        print(f"\n[RESULT] " + "\n".join(msg_parts))


if __name__ == "__main__":
    asyncio.run(main())


