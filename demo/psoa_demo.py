#!/usr/bin/env python3
"""
PSOA（并行自优化架构）原型验证脚本
四线程并行：G(生成) → R(审查) + L(放行) 并行 → I(代际蒸馏)
使用 Agnes AI API (兼容OpenAI格式) 模拟四线程角色分离
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI

# ===== 参数区 =====
result_mode = sys.argv[1] if len(sys.argv) > 1 else "notify"
question = sys.argv[2] if len(sys.argv) > 2 else "如果所有的猫都是动物，所有的动物都是生物，那么所有的猫都是生物吗？请详细解释你的推理过程。"
max_rounds = int(sys.argv[3]) if len(sys.argv) > 3 else "5"
task_type = sys.argv[4] if len(sys.argv) > 4 else "推理"

# ===== 输出目录 =====
OUTPUT_DIR = "/app/data/所有对话/主对话/PSOA/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Agnes AI API 配置 =====
API_KEY = os.environ.get("AGNES_API_KEY", "sk-KhMlcUhJIlwajOuCrDeHMbZhX8rlFFQ32QQS77ecP6JpCh1r")
BASE_URL = "https://apihub.agnes-ai.com/v1"
MODEL = "agnes-2.0-flash"

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ===== 四线程 System Prompt（严格论文定义） =====

G_SYSTEM = """你是PSOA架构的【G线程·生成线程】。
你的职责：接收任务描述、当前策略提示、审查历史，生成候选输出。
你必须：
1. 严格遵循策略提示中的方法论
2. 充分吸收审查历史中的修正建议
3. 输出完整的候选答案，不做自我审查
4. 用中文回答"""

R_SYSTEM = """你是PSOA架构的【R线程·审查线程】。
你的职责：对G线程的候选输出进行纯负反馈审查。
你必须且只能找出缺陷，输出结构化审查意见：
1. 错误类型：逻辑错误/事实错误/遗漏/不一致/表述模糊
2. 严重程度：致命(阻碍结论成立)/严重(影响推理质量)/轻微(可优化但不影响正确性)
3. 修正建议：具体的修改方向

关键规则：
- 你只负责找缺陷，不负责确认内容有多好
- 如果候选输出完美无缺，也要指出可以进一步优化的方向
- 必须输出结构化格式
- 用中文回答

致命判据（必须严格执行）：
- 事实性错误一律标为致命——事实是论证的根基，事实错了结论就站不住
- 逻辑矛盾/自相矛盾一律标为致命——与L线程的自洽性检验形成对冲
- 关键前提遗漏导致结论无法推出，一律标为致命
- 只有纯表述优化、措辞建议才可标为轻微
- 不要因为"核心结论碰巧对了"就降低严重程度——过程错误即使结论正确也是致命的"""

L_SYSTEM = """你是PSOA架构的【L线程·放行线程】。
你的判据是"这样说对了吗"——检验逻辑自洽性。
你必须从三个维度检验，并给出明确判断：

1. 内部一致性：推理过程中各步骤之间是否自洽，有无自相矛盾
2. 因果闭合性：因果链是否完整，前提是否能推出结论，有无逻辑跳跃
3. 语义完整性：关键概念是否定义清晰，语义是否完整无歧义

输出格式：
- 内部一致性: 通过/未通过 + 说明
- 因果闭合性: 通过/未通过 + 说明
- 语义完整性: 通过/未通过 + 说明
- 放行信号: 0(不放行) 或 1(放行)
- 判断理由: 一句话总结

关键规则：
- 你只判断逻辑自洽性，不做质量评分
- 三个维度全部通过才放行
- 用中文回答"""

I_SYSTEM = """你是PSOA架构的【I线程·代际蒸馏引擎】。
你的职责体现代际生命周期：蒸馏→繁衍→自消亡。

1. 蒸馏：从本轮交互中提取经验教训
2. 繁衍：将经验蒸馏为下一代策略提示
3. 自消亡：本轮策略已融入新策略，旧策略使命终结

输出格式：
- 本轮经验：从G/R/L三个线程的交互中提炼的关键经验
- 策略更新：哪些策略被保留、哪些被修正、哪些被新增
- 下一代策略：完整的下一代策略提示文本（G线程下一轮将使用此策略）

关键规则：
- 策略提示必须具体可执行，不要泛泛而谈
- 要体现经验的累积性，每一代策略应比上一代更精准
- 用中文回答"""


async def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """调用 Agnes AI API"""
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[API调用失败: {e}]"


async def g_thread(question: str, strategy: str, review_history: str) -> str:
    """G线程：生成候选输出"""
    user_prompt = f"""任务：{question}

当前策略提示：
{strategy}

审查历史（请充分吸收修正建议）：
{review_history if review_history else '（首轮，暂无审查历史）'}

请根据策略提示生成候选输出。"""
    return await call_llm(G_SYSTEM, user_prompt, temperature=0.8)


async def r_thread(candidate: str) -> str:
    """R线程：纯负反馈审查"""
    user_prompt = f"""请对以下候选输出进行纯负反馈审查：

---
{candidate}
---

请严格按格式输出：错误类型、严重程度、修正建议。"""
    return await call_llm(R_SYSTEM, user_prompt, temperature=0.3)


async def l_thread(candidate: str, question: str) -> str:
    """L线程：逻辑自洽性放行检验"""
    user_prompt = f"""原始问题：{question}

候选输出：
{candidate}

请检验此候选输出的逻辑自洽性（"这样说对了吗"），从内部一致性、因果闭合性、语义完整性三个维度判断，给出放行信号。"""
    return await call_llm(L_SYSTEM, user_prompt, temperature=0.2)


async def i_thread(question: str, strategy: str, candidate: str, review: str, let_through: str, round_num: int) -> str:
    """I线程：代际蒸馏引擎"""
    user_prompt = f"""任务：{question}

当前代（第{round_num}代）策略提示：
{strategy}

本轮G线程候选输出：
{candidate}

本轮R线程审查意见：
{review}

本轮L线程放行判断：
{let_through}

请执行代际蒸馏：提取经验→蒸馏策略→注入下一代→自消亡。输出完整的下一代策略提示。"""
    return await call_llm(I_SYSTEM, user_prompt, temperature=0.5)


def parse_let_through_signal(l_result: str) -> tuple:
    """从L线程结果中解析放行信号"""
    signal = 0
    # 尝试匹配放行信号
    for line in l_result.split('\n'):
        line_lower = line.lower()
        if '放行信号' in line:
            if '1' in line or '✓' in line or '通过' in line.split('：')[-1].split(':')[0] if '：' in line or ':' in line else '':
                signal = 1
            break
    # 更宽泛的匹配
    if signal == 0:
        if '放行信号: 1' in l_result or '放行信号：1' in l_result:
            signal = 1
        elif '放行信号: ✓' in l_result or '放行信号：✓' in l_result:
            signal = 1
    return signal


def has_fatal_error(r_result: str) -> bool:
    """检查R线程是否发现致命错误（含代码层强制升级）"""
    # 1. R线程自己标了致命
    if '致命' in r_result and ('严重程度' in r_result or '致命' in r_result):
        return True
    # 2. 代码层强制升级：事实错误一律视为致命，不管R标什么严重程度
    if '事实错误' in r_result or '事实性错误' in r_result:
        return True
    # 3. 代码层强制升级：逻辑矛盾/自相矛盾一律视为致命
    if '逻辑矛盾' in r_result or '自相矛盾' in r_result or '自相矛盾' in r_result:
        return True
    return False


def format_round_report(round_num: int, max_r: int, strategy: str, candidate: str,
                        review: str, let_through: str, i_result: str,
                        signal: int, elapsed: float) -> str:
    """格式化单轮中文过程报告"""
    # 截断过长内容用于展示
    def trunc(text, limit=500):
        if len(text) > limit:
            return text[:limit] + f"...（共{len(text)}字）"
        return text

    signal_display = "✓" if signal == 1 else "✗"

    report = f"""
{'='*35}
第 {round_num} 轮 / 最多 {max_r} 轮  |  耗时 {elapsed:.1f}s
{'='*35}

【G线程·生成】
  策略提示: {trunc(strategy, 300)}
  候选输出: {trunc(candidate)}

【R线程·审查】
  {trunc(review)}

【L线程·放行】
  {trunc(let_through)}
  放行信号: {signal_display}

【I线程·代际蒸馏】
  {trunc(i_result)}

{'='*35}
"""
    return report


async def main():
    print(f"[参数] result_mode={result_mode}, question={question[:50]}..., max_rounds={max_rounds}, task_type={task_type}")

    # ===== 初始化策略提示 =====
    strategy = f"""【初始策略·第0代】
任务类型: {task_type}
方法论: 
1. 仔细分析问题，识别核心前提与结论
2. 逐步推理，确保每一步都有充分依据
3. 验证结论是否由前提逻辑推出
4. 检查是否有遗漏的中间环节
5. 用清晰的语言组织答案"""

    review_history = ""
    all_reports = []
    round_signals = []
    experience_log = []
    final_answer = ""
    converged = False

    start_time = time.time()

    for round_num in range(1, max_rounds + 1):
        round_start = time.time()
        print(f"\n>>> 开始第 {round_num} 轮迭代...")

        # ===== G线程：生成候选输出 =====
        print(f"  [G线程] 正在生成候选输出...")
        candidate = await g_thread(question, strategy, review_history)
        print(f"  [G线程] 生成完毕，长度={len(candidate)}")

        # ===== R线程 + L线程：并行执行 =====
        print(f"  [R线程+L线程] 并行执行中...")
        r_result, l_result = await asyncio.gather(
            r_thread(candidate),
            l_thread(candidate, question),
        )
        print(f"  [R线程] 审查完毕，长度={len(r_result)}")
        print(f"  [L线程] 放行判断完毕，长度={len(l_result)}")

        # ===== 解析放行信号 =====
        signal = parse_let_through_signal(l_result)
        fatal = has_fatal_error(r_result)
        round_signals.append(signal)

        print(f"  [判断] 放行信号={signal}, 致命错误={fatal}")

        # ===== I线程：代际蒸馏 =====
        print(f"  [I线程] 代际蒸馏中...")
        i_result = await i_thread(question, strategy, candidate, r_result, l_result, round_num)
        print(f"  [I线程] 蒸馏完毕，长度={len(i_result)}")

        # ===== 更新策略提示（I线程的输出） =====
        # 从I线程结果中提取"下一代策略"
        new_strategy = strategy  # 默认保持
        if '下一代策略' in i_result:
            # 尝试提取下一代策略内容
            parts = i_result.split('下一代策略')
            if len(parts) > 1:
                strategy_text = parts[1].lstrip('：:').strip()
                # 取到下一个小节之前
                for sep in ['\n\n---', '\n\n【', '\n\n策略更新']:
                    if sep in strategy_text:
                        strategy_text = strategy_text[:strategy_text.index(sep)]
                if len(strategy_text) > 50:  # 确保提取到了有意义的内容
                    new_strategy = strategy_text

        # ===== 更新审查历史 =====
        review_history += f"\n第{round_num}轮审查意见：{r_result[:500]}"
        if len(review_history) > 3000:
            review_history = review_history[-2000:]

        # ===== 记录经验 =====
        experience_log.append({
            "round": round_num,
            "signal": signal,
            "fatal": fatal,
            "strategy_len": len(new_strategy),
        })

        # ===== 格式化报告 =====
        elapsed = time.time() - round_start
        report = format_round_report(
            round_num, max_rounds, strategy, candidate,
            r_result, l_result, i_result, signal, elapsed
        )
        all_reports.append(report)
        print(report)

        # ===== 收敛判断 =====
        if signal == 1 and not fatal:
            final_answer = candidate
            converged = True
            print(f">>> ✓ 第 {round_num} 轮收敛！L线程放行且无致命错误。")
            break
        else:
            strategy = new_strategy
            print(f">>> ✗ 第 {round_num} 轮未收敛，进入下一代...")

    total_time = time.time() - start_time

    # ===== 最终摘要 =====
    signal_display = " → ".join([f"✓" if s == 1 else "✗" for s in round_signals])
    final_summary = f"""
{'='*35}
PSOA 最终摘要
{'='*35}

问题: {question}
任务类型: {task_type}
总轮数: {len(round_signals)} / {max_rounds}
是否收敛: {'是 ✓' if converged else '否 ✗（已达最大轮数）'}
总耗时: {total_time:.1f}s

各轮放行信号变化:
  {signal_display}

经验积累总结:
"""

    for exp in experience_log:
        sig = "✓" if exp["signal"] == 1 else "✗"
        fatal_mark = " [致命]" if exp["fatal"] else ""
        final_summary += f"  第{exp['round']}轮: 放行={sig}{fatal_mark}, 策略长度={exp['strategy_len']}\n"

    if final_answer:
        final_summary += f"""
最终答案:
{final_answer}
"""

    print(final_summary)

    # ===== 保存完整日志 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"psoa_log_{timestamp}.md"
    log_path = os.path.join(OUTPUT_DIR, log_filename)

    full_log = f"""# PSOA 并行自优化架构 - 完整过程日志

## 基本信息
- 时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 问题: {question}
- 任务类型: {task_type}
- 最大轮数: {max_rounds}
- 实际轮数: {len(round_signals)}
- 是否收敛: {'是' if converged else '否'}
- 总耗时: {total_time:.1f}s

## 迭代过程

{''.join(all_reports)}

## 最终摘要
{final_summary}
"""

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(full_log)
    print(f"\n[日志] 完整日志已保存: {log_path}")

    # ===== 提交结果 =====
    # 构建提交消息
    msg_parts = [
        f"PSOA原型验证完成",
        f"问题: {question[:60]}{'...' if len(question) > 60 else ''}",
        f"收敛: {'是' if converged else '否'} | 轮数: {len(round_signals)}/{max_rounds} | 耗时: {total_time:.1f}s",
        f"放行信号变化: {signal_display}",
    ]
    if converged and final_answer:
        msg_parts.append(f"最终答案摘要: {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
    else:
        msg_parts.append("未收敛，建议增加最大迭代轮数或调整策略。")

    # 使用 CodeActSDK 提交
    try:
        from codeact_sdk import CodeActSDK
        sdk = CodeActSDK()
        await sdk.submit_result(
            result_mode=result_mode if result_mode != "auto" else "notify",
            status="success",
            message="\n".join(msg_parts),
            data={
                "converged": converged,
                "rounds": len(round_signals),
                "max_rounds": max_rounds,
                "signals": round_signals,
                "log_file": log_path,
                "final_answer": final_answer[:500] if final_answer else "",
            },
        )
    except Exception as e:
        print(f"[提交结果失败] {e}")
        # 备用：直接输出结果
        print(f"\n[RESULT] " + "\n".join(msg_parts))


if __name__ == "__main__":
    asyncio.run(main())
