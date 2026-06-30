#!/usr/bin/env python3
"""
PSOA v2.2 - 结构化升级版（I线程深度蒸馏 + Claude 审查修复）
基于 v1 原型 + ChatGPT/Grok 评审意见改进 + Claude 审查bug修复 + I线程蒸馏升级：
  1. R/L/I 三线程全部 JSON 结构化输出 + extract_json 容错
  2. I 线程策略进化使用 retained/modified/new 结构
  3. review_history 改为滑动窗口 + 压缩摘要，不再全文累积
  4. L 线程 fatal_reconciliation：审计字段，R 标致命时 L 必须正面回应
     （设计决策：R 的致命判定为一票否决，fatal_reconciliation 不影响收敛控制流，
      仅用于审计留痕和复盘 R 是否判过严。详见收敛判断逻辑。）
  5. 保留中文过程日志（v1 核心优点）
  6. Config 类整理参数
  7. [v2.1] has_fatal_error 加 isinstance 类型保护，防止 errors 字段类型异常崩溃
  8. [v2.1] call_llm 异常捕获扩宽，网络超时/None响应不再捅穿 main()
  9. [v2.1] has_fatal_error 对"逻辑错误+严重"不再强制升级为致命，与 R_SYSTEM 对齐
  10. [v2.1] 未收敛时保存最后一轮候选答案供参考
  11. [v2.1] strategy 长度上限检查，防止 token 失控
  12. [v2.1] 移除硬编码 API key，仅从环境变量读取
  13. [v2.2] I线程升级为深度蒸馏：根因诊断→教训转化→策略进化→自验证四步法
  14. [v2.2] I线程输出新增 round_diagnosis/key_lessons/expected_improvement 字段
  15. [v2.2] I线程策略进化要求：必须含检查清单+决策流程，禁止泛泛而谈
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from openai import AsyncOpenAI

# ==================== 配置区 ====================
class Config:
    RESULT_MODE = sys.argv[1] if len(sys.argv) > 1 else "notify"
    QUESTION = sys.argv[2] if len(sys.argv) > 2 else "如果所有的猫都是动物，所有的动物都是生物，那么所有的猫都是生物吗？请详细解释你的推理过程。"
    MAX_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    TASK_TYPE = sys.argv[4] if len(sys.argv) > 4 else "推理"
    
    OUTPUT_DIR = Path("/app/data/所有对话/主对话/PSOA/output")
    API_KEY = os.environ.get("AGNES_API_KEY", "")
    BASE_URL = "https://apihub.agnes-ai.com/v1"
    MODEL = "agnes-2.0-flash"
    
    # 滑动窗口：最多保留最近N轮审查历史
    REVIEW_WINDOW = 3
    # 审查历史压缩摘要长度上限
    REVIEW_SUMMARY_MAX = 800
    # 策略提示长度上限（防止 token 失控）
    STRATEGY_MAX_LEN = 3000


Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
client = AsyncOpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)


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
- "严重"仅用于：推理过程有瑕疵但不影响结论成立的情况
- errors 字段必须是数组，即使没有错误也要输出空数组 []"""

L_SYSTEM = """你是PSOA架构的【L线程·放行线程】，判据是"这样说对了吗"——检验逻辑自洽性。

请严格按以下JSON格式输出（不要添加JSON以外的文字）：

{
  "internal_consistency": {"passed": true/false, "comment": "推理各步骤之间是否自洽"},
  "causal_closure": {"passed": true/false, "comment": "因果链是否完整，前提能否推出结论"},
  "semantic_completeness": {"passed": true/false, "comment": "关键概念是否定义清晰，语义是否完整"},
  "fatal_reconciliation": {"comment": "审计字段：如果R线程标记了致命错误，说明你为什么仍认为逻辑自洽（或承认R的致命发现成立）。注意：此字段仅用于审计留痕，不影响收敛判定——R标致命时本轮必定不收敛。如R未标致命，填'不适用'"},
  "let_through": 1,
  "summary": "一句话最终判断理由"
}

关键规则：
- 三个维度全部通过才可 let_through = 1
- fatal_reconciliation 是审计留痕字段：R标致命时你必须正面回应，但即使你论证了R判过严，本轮仍不会收敛（R的致命判定为一票否决）
- 你只判断逻辑自洽性，不做质量评分
- 用中文填写comment和summary"""

I_SYSTEM = """你是PSOA架构的【I线程·代际蒸馏引擎】。
你的职责是进行真正有价值的代际蒸馏：从本轮交互中提炼可传承的智慧，并转化为下一代更强的策略。体现代际生命周期：蒸馏→繁衍→自消亡。

请严格按以下JSON格式输出：

{
  "round_diagnosis": "本轮最核心的问题是什么？（一句话诊断根因，而非表面症状）",
  "key_lessons": ["从G/R/L交互中提炼的2-4条具体、可操作的经验教训"],
  "strategy_updates": {
    "retained": ["本轮验证有效、继续保留的策略要点"],
    "modified": ["需要修正的策略 + 具体修正方向"],
    "new": ["本轮新发现的、能显著提升能力的策略或检查清单"]
  },
  "next_strategy": "完整的下一代策略提示文本",
  "expected_improvement": "这一代相比上一代，具体会在哪些方面变强？（例如：更擅长排除极端情况、更能捕捉隐含假设等）"
}

蒸馏时必须严格遵循以下四步思考过程：

1. **根因诊断**：本轮G输出的根本弱点是什么？R和L分别暴露了什么系统性问题？不要只描述表面症状，要找到根因。

2. **教训转化**：把问题转化为具体的、可执行的防御机制或检查清单。绝对不能输出"要更仔细"这种空话——每条教训必须能直接变成下一步的检查步骤。

3. **策略进化**：下一代策略必须比上一代更精准、更具操作性。要加入本轮新发现的检查步骤，去掉被证明无效的旧策略。

4. **自验证**：思考"用这个新策略，下次遇到类似问题会更好吗？哪里还有可能失效？"如果发现新策略仍有明显盲区，在strategy_updates.new中补充防御措施。

下一代策略要求：
- 必须包含明确的检查清单和决策流程
- 针对本轮暴露的弱点设计针对性改进
- 语言要具体可执行，不要泛泛而谈
- 整体长度适中，但信息密度更高"""


# ==================== 工具函数 ====================

async def call_llm(system: str, user: str, temperature: float = 0.7, retries: int = 2) -> str:
    """调用 Agnes AI API，带重试，异常捕获覆盖所有情况"""
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
            content = response.choices[0].message.content
            if content is None:
                if attempt == retries:
                    return '[API返回空内容]'
                await asyncio.sleep(1 * (attempt + 1))
                continue
            return content.strip()
        except Exception as e:
            if attempt == retries:
                return f'[API调用失败: {type(e).__name__}: {e}]'
            await asyncio.sleep(1 * (attempt + 1))
    return '[API调用失败: 超过最大重试次数]'


def extract_json(text: str) -> Dict:
    """从文本中提取JSON，容错处理"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return {}


def _safe_errors(errors_field) -> list:
    """类型安全地获取 errors 列表，防止 LLM 输出非数组类型导致崩溃"""
    if isinstance(errors_field, list):
        return errors_field
    if isinstance(errors_field, str):
        # LLM 可能输出 "未发现明显问题" 之类的字符串
        return []
    return []


def parse_r_result(raw: str) -> Dict:
    """解析R线程输出，兼容JSON和纯文本，带类型保护"""
    j = extract_json(raw)
    if j:
        j["errors"] = _safe_errors(j.get("errors", []))
        return j
    # 降级：纯文本解析
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


def parse_l_result(raw: str) -> Dict:
    """解析L线程输出，兼容JSON和纯文本，带类型保护"""
    j = extract_json(raw)
    if j:
        # 类型保护各维度字段
        for key in ("internal_consistency", "causal_closure", "semantic_completeness", "fatal_reconciliation"):
            if not isinstance(j.get(key), dict):
                j[key] = {"passed": False, "comment": str(j.get(key, ""))}
        # 确保 let_through 是 int
        lt = j.get("let_through", 0)
        j["let_through"] = 1 if lt in (1, True, "1") else 0
        return j
    # 降级：从纯文本中提取放行信号
    signal = 0
    for line in raw.split('\n'):
        if '放行信号' in line or 'let_through' in line:
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


def has_fatal_error(r_parsed: Dict) -> bool:
    """检查R线程是否发现致命错误
    
    设计决策：R 的致命判定为一票否决，L 的 fatal_reconciliation 不影响此判定。
    这不是 bug——PSOA 的收敛条件是 L 放行 + R 无致命，双重闸门。
    fatal_reconciliation 仅用于审计留痕，帮助事后复盘 R 是否判过严。
    """
    errors = _safe_errors(r_parsed.get("errors", []))
    for err in errors:
        if not isinstance(err, dict):
            continue
        severity = str(err.get("severity", ""))
        err_type = str(err.get("type", ""))
        desc = str(err.get("description", ""))
        # 1. R自己标了致命
        if severity == "致命":
            return True
        # 2. 代码层强制升级：事实错误一律致命（与 R_SYSTEM 对齐）
        if err_type == "事实错误" and severity in ("严重", "致命"):
            return True
        # 3. 代码层强制升级：逻辑矛盾/自相矛盾一律致命（与 R_SYSTEM 对齐）
        if any(kw in desc for kw in ["逻辑矛盾", "自相矛盾"]):
            return True
    # 4. 纯文本降级时的检查
    overall = str(r_parsed.get("overall_assessment", ""))
    if '致命' in overall:
        return True
    return False


def compress_review_history(history_entries: list, window: int = Config.REVIEW_WINDOW) -> str:
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


async def l_thread(candidate: str, question: str, r_parsed: Dict) -> str:
    """L线程：逻辑自洽性放行检验
    
    注意：v2 因为 L 需要 r_parsed 来构建 fatal_reconciliation 提示，
    R 和 L 从并行变为串行。这是功能权衡——L 需要看到 R 的致命判定才能正面回应。
    如需恢复并行，可改为：先并行跑 R+L(无对冲)，R 标致命时再补一次 L 调用。
    """
    # 构建R线程致命信息，让L正面回应
    fatal_errors = []
    for err in _safe_errors(r_parsed.get("errors", [])):
        if isinstance(err, dict) and err.get("severity") == "致命":
            fatal_errors.append(f"- [{err.get('type')}] {err.get('description')}")
    
    r_fatal_info = ""
    if fatal_errors:
        r_fatal_info = f"""

⚠️ R线程标记了以下致命错误（你必须在fatal_reconciliation字段中正面回应——这是审计留痕，不影响收敛判定）：
{chr(10).join(fatal_errors)}"""

    user_prompt = f"""原始问题：{question}

候选输出：
{candidate}
{r_fatal_info}

请检验此候选输出的逻辑自洽性，输出严格JSON格式。"""
    return await call_llm(L_SYSTEM, user_prompt, temperature=0.2)


async def i_thread(question: str, strategy: str, candidate: str, 
                   r_result: str, l_result: str, round_num: int) -> Dict:
    """I线程：代际蒸馏引擎"""
    user_prompt = f"""任务：{question}

当前代（第{round_num}代）策略提示：
{strategy}

本轮G线程候选输出：
{candidate}

本轮R线程审查意见：
{r_result}

本轮L线程放行判断：
{l_result}

轮次：{round_num}

请执行代际蒸馏，输出严格JSON格式。"""
    result = await call_llm(I_SYSTEM, user_prompt, temperature=0.5)
    parsed = extract_json(result)
    if not parsed:
        parsed = {
            "round_diagnosis": result[:100],
            "key_lessons": [],
            "strategy_updates": {"retained": [], "modified": [], "new": []},
            "next_strategy": strategy,
            "expected_improvement": ""
        }
    # 类型保护 round_diagnosis / key_lessons / expected_improvement
    if not isinstance(parsed.get("round_diagnosis"), str):
        parsed["round_diagnosis"] = str(parsed.get("round_diagnosis", ""))
    if not isinstance(parsed.get("key_lessons"), list):
        parsed["key_lessons"] = []
    if not isinstance(parsed.get("expected_improvement"), str):
        parsed["expected_improvement"] = str(parsed.get("expected_improvement", ""))
    # 类型保护 strategy_updates
    su = parsed.get("strategy_updates", {})
    if not isinstance(su, dict):
        su = {"retained": [], "modified": [], "new": []}
    for key in ("retained", "modified", "new"):
        if not isinstance(su.get(key), list):
            su[key] = []
    parsed["strategy_updates"] = su
    return parsed


# ==================== 过程报告 ====================

def format_round_report(round_num: int, max_r: int, strategy: str, candidate: str,
                        r_parsed: Dict, l_parsed: Dict, i_parsed: Dict,
                        signal: int, fatal: bool, elapsed: float) -> str:
    """格式化单轮中文过程报告（保留v1核心优点）"""
    def trunc(text, limit=500):
        if isinstance(text, (dict, list)):
            text = json.dumps(text, ensure_ascii=False, indent=2)
        if len(str(text)) > limit:
            return str(text)[:limit] + f"...（共{len(str(text))}字）"
        return str(text)

    signal_display = "✓ 放行" if signal == 1 else "✗ 不放行"
    fatal_display = "⚠️ 有致命错误" if fatal else "无致命错误"

    # R线程结构化展示
    r_errors = _safe_errors(r_parsed.get("errors", []))
    r_summary = r_parsed.get("overall_assessment", "无")
    r_display = f"  审查总结: {r_summary}\n"
    for i, err in enumerate(r_errors[:5], 1):
        if not isinstance(err, dict):
            continue
        r_display += f"  错误{i}: [{err.get('severity','?')}] {err.get('type','?')} - {trunc(err.get('description',''), 150)}\n"
        if err.get('suggestion'):
            r_display += f"         建议: {trunc(err['suggestion'], 100)}\n"

    # L线程结构化展示
    ic = l_parsed.get("internal_consistency", {})
    cc = l_parsed.get("causal_closure", {})
    sc = l_parsed.get("semantic_completeness", {})
    fr = l_parsed.get("fatal_reconciliation", {})
    l_display = f"  内部一致性: {'✓' if ic.get('passed') else '✗'} {ic.get('comment', '')}\n"
    l_display += f"  因果闭合性: {'✓' if cc.get('passed') else '✗'} {cc.get('comment', '')}\n"
    l_display += f"  语义完整性: {'✓' if sc.get('passed') else '✗'} {sc.get('comment', '')}\n"
    l_display += f"  致命对冲(审计): {fr.get('comment', '不适用')}\n"
    l_display += f"  判断理由: {l_parsed.get('summary', '')}\n"

    # I线程结构化展示
    su = i_parsed.get("strategy_updates", {})
    i_display = f"  根因诊断: {trunc(i_parsed.get('round_diagnosis', ''), 300)}\n"
    kl = i_parsed.get("key_lessons", [])
    if kl:
        i_display += f"  关键教训: {kl[:4]}\n"
    i_display += f"  保留策略: {su.get('retained', [])}\n"
    i_display += f"  修正策略: {su.get('modified', [])}\n"
    i_display += f"  新增策略: {su.get('new', [])}\n"
    ei = i_parsed.get("expected_improvement", "")
    if ei:
        i_display += f"  预期改进: {trunc(ei, 200)}\n"

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


# ==================== 主逻辑 ====================

async def main():
    print(f"[PSOA v2.2] result_mode={Config.RESULT_MODE}, question={Config.QUESTION[:50]}..., max_rounds={Config.MAX_ROUNDS}, task_type={Config.TASK_TYPE}")

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
    best_candidate = ""  # 未收敛时保存最像样的候选
    best_candidate_round = 0
    converged = False

    start_time = time.time()

    for round_num in range(1, Config.MAX_ROUNDS + 1):
        round_start = time.time()
        print(f"\n>>> 第 {round_num} 轮开始...")

        # G线程
        print(f"  [G] 生成中...")
        candidate = await g_thread(Config.QUESTION, strategy, compress_review_history(review_entries))
        print(f"  [G] 完成, len={len(candidate)}")

        # R线程
        print(f"  [R] 审查中...")
        r_raw = await r_thread(candidate)
        r_parsed = parse_r_result(r_raw)
        print(f"  [R] 完成, errors={len(_safe_errors(r_parsed.get('errors', [])))}")

        # L线程（串行：L需要R的解析结果来构建fatal_reconciliation提示）
        print(f"  [L] 放行判断中...")
        l_raw = await l_thread(candidate, Config.QUESTION, r_parsed)
        l_parsed = parse_l_result(l_raw)
        print(f"  [L] 完成, let_through={l_parsed.get('let_through', 0)}")

        # 解析信号
        signal = l_parsed.get("let_through", 0)
        fatal = has_fatal_error(r_parsed)
        round_signals.append(signal)
        print(f"  [判断] 放行={signal}, 致命={fatal}")

        # I线程
        print(f"  [I] 代际蒸馏中...")
        i_parsed = await i_thread(Config.QUESTION, strategy, candidate, r_raw, l_raw, round_num)
        print(f"  [I] 完成")

        # 更新策略
        new_strategy = i_parsed.get("next_strategy", strategy)
        if len(new_strategy) < 50:
            new_strategy = strategy  # I输出异常时保持原策略
        # 策略长度上限检查
        if len(new_strategy) > Config.STRATEGY_MAX_LEN:
            print(f"  [警告] 策略过长({len(new_strategy)}字)，截断至{Config.STRATEGY_MAX_LEN}字")
            new_strategy = new_strategy[:Config.STRATEGY_MAX_LEN]

        # 更新审查历史（滑动窗口）
        r_entry = f"第{round_num}轮: " + json.dumps(_safe_errors(r_parsed.get("errors", [])), ensure_ascii=False)[:Config.REVIEW_SUMMARY_MAX]
        review_entries.append(r_entry)

        # 保存最像样的候选（优先：L放行 > 无致命 > 最新）
        if signal == 1:
            best_candidate = candidate
            best_candidate_round = round_num
        elif not fatal and not best_candidate:
            best_candidate = candidate
            best_candidate_round = round_num

        # 记录经验
        experience_log.append({
            "round": round_num,
            "signal": signal,
            "fatal": fatal,
            "errors_count": len(_safe_errors(r_parsed.get("errors", []))),
            "fatal_reconciliation": l_parsed.get("fatal_reconciliation", {}).get("comment", "") if isinstance(l_parsed.get("fatal_reconciliation"), dict) else str(l_parsed.get("fatal_reconciliation", "")),
            "round_diagnosis": i_parsed.get("round_diagnosis", ""),
            "key_lessons": i_parsed.get("key_lessons", []),
            "strategy_updates": i_parsed.get("strategy_updates", {}),
            "expected_improvement": i_parsed.get("expected_improvement", ""),
        })

        # 格式化报告
        elapsed = time.time() - round_start
        report = format_round_report(
            round_num, Config.MAX_ROUNDS, strategy, candidate,
            r_parsed, l_parsed, i_parsed, signal, fatal, elapsed
        )
        all_reports.append(report)
        print(report)

        # 收敛判断
        if signal == 1 and not fatal:
            final_answer = candidate
            converged = True
            print(f">>> ✓ 第 {round_num} 轮收敛！L放行 + 无致命错误。")
            break
        else:
            strategy = new_strategy
            reason = "R标致命" if fatal else "L不放行"
            print(f">>> ✗ 第 {round_num} 轮未收敛（{reason}），进入下一代...")

    total_time = time.time() - start_time

    # 最终摘要
    signal_display = " → ".join(["✓" if s == 1 else "✗" for s in round_signals])
    final_summary = f"""
{'='*40}
PSOA v2.2 最终摘要
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
        diag = exp.get("round_diagnosis", "")
        diag_display = f" | 诊断: {diag[:60]}" if diag else ""
        su = exp.get("strategy_updates", {})
        su_display = f" 保留{len(su.get('retained',[]))} 修正{len(su.get('modified',[]))} 新增{len(su.get('new',[]))}"
        ei = exp.get("expected_improvement", "")
        ei_display = f" | 预期: {ei[:60]}" if ei else ""
        final_summary += f"  第{exp['round']}轮: 放行={sig}{fatal_mark} 错误数={exp['errors_count']}{fr_display}{diag_display}{su_display}{ei_display}\n"

    if final_answer:
        final_summary += f"\n最终答案:\n{final_answer}\n"
    elif best_candidate:
        final_summary += f"\n最佳候选（未通过审查，仅供参考·来自第{best_candidate_round}轮）:\n{best_candidate}\n"

    print(final_summary)

    # 保存日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"psoa_v2_log_{timestamp}.md"
    log_path = Config.OUTPUT_DIR / log_filename

    full_log = f"""# PSOA v2.2 并行自优化架构 - 完整过程日志

## 基本信息
- 版本: v2.2（I线程深度蒸馏升级 + Claude 审查修复）
- 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 问题: {Config.QUESTION}
- 任务类型: {Config.TASK_TYPE}
- 最大轮数: {Config.MAX_ROUNDS}
- 实际轮数: {len(round_signals)}
- 是否收敛: {'是' if converged else '否'}
- 总耗时: {total_time:.1f}s

## v2.2 改进点
1. R/L/I 三线程 JSON 结构化输出 + extract_json 容错
2. I 线程策略进化 retained/modified/new 结构
3. review_history 滑动窗口 + 压缩摘要
4. L 线程 fatal_reconciliation：审计字段，R 标致命时 L 必须正面回应
5. 保留中文过程日志
6. Config 类整理参数
7. [v2.1] isinstance 类型保护，防止 errors 字段类型异常崩溃
8. [v2.1] call_llm 异常捕获扩宽，网络超时/None响应不再捅穿
9. [v2.1] has_fatal_error 对"逻辑错误+严重"不再强制升级为致命
10. [v2.1] 未收敛时保存最佳候选供参考
11. [v2.1] strategy 长度上限检查
12. [v2.1] 移除硬编码 API key
13. [v2.2] I线程升级为深度蒸馏：根因诊断→教训转化→策略进化→自验证四步法
14. [v2.2] I线程输出新增 round_diagnosis/key_lessons/expected_improvement
15. [v2.2] I线程策略要求：必须含检查清单+决策流程，禁止泛泛而谈

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
        f"PSOA v2.2 原型验证完成",
        f"问题: {Config.QUESTION[:60]}{'...' if len(Config.QUESTION) > 60 else ''}",
        f"收敛: {'是' if converged else '否'} | 轮数: {len(round_signals)}/{Config.MAX_ROUNDS} | 耗时: {total_time:.1f}s",
        f"信号: {signal_display}",
    ]
    if converged and final_answer:
        msg_parts.append(f"答案摘要: {final_answer[:200]}")
    elif best_candidate:
        msg_parts.append(f"最佳候选(未通过审查): {best_candidate[:200]}")
    else:
        msg_parts.append("未收敛，无候选答案。")

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
                "best_candidate": best_candidate[:500] if best_candidate else "",
            },
        )
    except Exception as e:
        print(f"[提交失败] {e}")
        print(f"\n[RESULT] " + "\n".join(msg_parts))


if __name__ == "__main__":
    asyncio.run(main())
