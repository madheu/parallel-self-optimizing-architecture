#!/usr/bin/env python3
'''
PSOA + Verifier Integration
===========================
Extends psoa_demo_v2.py with the external Verifier layer.

Usage:
    python psoa_verifier.py [result_mode] [question] [max_rounds] [task_type]
'''

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from psoa_demo_v2 import (
    Config,
    G_SYSTEM, R_SYSTEM, L_SYSTEM, I_SYSTEM,
    call_llm, call_llm_detailed, extract_json,
    parse_r_result, parse_l_result, has_fatal_error,
    candidate_similarity, compress_review_history,
    format_round_report,
    g_thread, r_thread, l_thread_raw, i_thread,
)
from openai import AsyncOpenAI
from verifier import (
    Verifier, Verdict, TrajectoryExporter,
    build_candidate_from_psoa,
)


async def run_psoa_with_verifier(
    question,
    task_type='推理',
    max_rounds=5,
    temperature=0.7,
    enable_llm_judge=True,
    verifier_output_dir=None,
):
    strategy = '''【第0代初始策略】
任务类型: {task_type}
方法论:
1. 仔细分析问题，识别核心前提与结论
2. 逐步推理，确保每一步都有充分依据
3. 验证结论是否由前提逻辑推出
4. 检查是否有遗漏的中间环节
5. 用清晰的语言组织答案'''.format(task_type=task_type)

    review_entries = []
    round_signals = []
    experience_log = []
    final_answer = ''
    converged = False
    last_passing_answer = ''
    total_input_tokens = 0
    total_output_tokens = 0

    start_time = time.time()

    verifier = Verifier(
        client=None if not enable_llm_judge else AsyncOpenAI(
            api_key=Config.API_KEY(),
            base_url=Config.BASE_URL,
        ),
        enable_llm_judge=enable_llm_judge,
    )

    output_dir = verifier_output_dir or str(Config.OUTPUT_DIR / 'verifier')
    exporter = TrajectoryExporter(output_dir)

    for round_num in range(1, max_rounds + 1):
        round_start = time.time()
        print(f'\n>>> 第 {round_num} 轮开始...')

        # === PSOA Core Loop ===
        print(f'  [G] 生成中...')
        candidate = await g_thread(
            question, strategy,
            compress_review_history(review_entries)
        )
        print(f'  [G] 完成, len={len(candidate)}')

        print(f'  [R] 审查中...')
        r_raw = await r_thread(candidate)
        r_parsed = parse_r_result(r_raw)
        print(f'  [R] 完成, errors={len(r_parsed.get("errors", []))}')

        print(f'  [L] 放行判断中...')
        l_raw = await l_thread_raw(candidate, question, r_raw)
        l_parsed = parse_l_result(l_raw)
        print(f'  [L] 完成, let_through={l_parsed.get("let_through", 0)}')

        signal = l_parsed.get('let_through', 0)
        fatal = has_fatal_error(r_parsed)
        round_signals.append(signal)
        print(f'  [判断] 放行={signal}, 致命={fatal}')

        print(f'  [I] 代际蒸馏中...')
        r_summary = json.dumps(r_parsed.get('errors', []), ensure_ascii=False, indent=2)[:600]
        l_summary = json.dumps(
            {k: v for k, v in l_parsed.items() if k != 'fatal_reconciliation'},
            ensure_ascii=False, indent=2
        )[:400]
        fr = l_parsed.get('fatal_reconciliation', {})
        if isinstance(fr, dict):
            l_summary += '\n致命对冲: ' + fr.get('comment', '')
        else:
            l_summary += '\n致命对冲: ' + str(fr)
        i_parsed = await i_thread(question, strategy, candidate, r_summary, l_summary, round_num)
        print(f'  [I] 完成')

        new_strategy = i_parsed.get('next_strategy', strategy)
        if len(new_strategy) < 50:
            new_strategy = strategy

        r_entry = '第' + str(round_num) + '轮: ' + json.dumps(
            r_parsed.get('errors', []), ensure_ascii=False
        )[:Config.REVIEW_SUMMARY_MAX]
        review_entries.append(r_entry)

        experience_log.append({
            'round': round_num,
            'signal': signal,
            'fatal': fatal,
            'errors_count': len(r_parsed.get('errors', [])),
            'fatal_reconciliation': (
                lambda v: v.get('comment', '') if isinstance(v, dict) else str(v)
            )(l_parsed.get('fatal_reconciliation', '')),
            'strategy_updates': i_parsed.get('strategy_updates', {}),
        })

        # === VERIFIER LAYER ===
        print(f'  [V] Verifier 验证中...')
        candidate_obj = build_candidate_from_psoa(
            question=question,
            candidate=candidate,
            round_num=round_num,
            strategy=strategy,
            r_parsed=r_parsed,
            l_parsed=l_parsed,
        )
        verifier_report = await verifier.verify(
            question=question,
            candidate=candidate,
            round_num=round_num,
            strategy=strategy,
            r_feedback=json.dumps(r_parsed, ensure_ascii=False)[:500],
            l_signal=signal,
            r_parsed=r_parsed,
            l_parsed=l_parsed,
        )
        print(verifier.format_report(verifier_report))
        print(f'  [V] Verdict: {verifier_report.verdict.value}')

        exporter.ingest(verifier_report, candidate_obj)

        elapsed = time.time() - round_start
        report = format_round_report(
            round_num, max_rounds, strategy, candidate,
            r_parsed, l_parsed, i_parsed, signal, fatal, elapsed
        )
        print(report)

        if verifier_report.verdict == Verdict.PASS and signal == 1 and not fatal:
            if not converged:
                final_answer = candidate
                last_passing_answer = candidate
                converged = True
                print(f'>>> V PASS 第 {round_num} 轮首次Verifier PASS，记录基准答案。')
            else:
                sim = candidate_similarity(last_passing_answer, candidate)
                if sim > 0.85:
                    final_answer = candidate
                    print(f'>>> V 收敛！第 {round_num} 轮Verifier PASS + L放行 + 答案稳定(相似度={sim:.2f})。')
                    break
                else:
                    print(f'>>> V 第 {round_num} 轮Verifier PASS但答案变化较大(相似度={sim:.2f})，继续进化...')
                    last_passing_answer = candidate
                    strategy = new_strategy
        else:
            strategy = new_strategy
            if verifier_report.verdict == Verdict.REJECT:
                print(f'>>> V 第 {round_num} 轮Verifier拒绝，进入下一代...')
            elif signal == 0:
                print(f'>>> V 第 {round_num} 轮L不放行，进入下一代...')
            elif fatal:
                print(f'>>> V 第 {round_num} 轮R标致命，进入下一代...')

    total_time = time.time() - start_time
    actual_rounds = len(round_signals)

    if not converged and round_signals:
        final_answer = candidate

    sft_path = exporter.export_sft_jsonl()
    dpo_path = exporter.export_dpo_jsonl()
    summary_path = exporter.export_summary()

    print(f'\n[Trajectory Export]')
    print(f'  SFT:  {sft_path} ({len(exporter.passed)} passed)')
    print(f'  DPO:  {dpo_path} (chosen={len(exporter.passed)} x rejected={len(exporter.rejected)})')
    print(f'  Summary: {summary_path}')

    return final_answer, converged, total_time, exporter


async def main():
    print(f'[PSOA+Verifier] question={Config.QUESTION[:50]}..., max_rounds={Config.MAX_ROUNDS}')

    final_answer, converged, total_time, exporter = await run_psoa_with_verifier(
        question=Config.QUESTION,
        task_type=Config.TASK_TYPE,
        max_rounds=Config.MAX_ROUNDS,
        temperature=0.7,
        enable_llm_judge=True,
    )

    print(f'\n{"=" * 50}')
    print(f'PSOA+Verifier 最终结果')
    print(f'{"=" * 50}')
    sig = '是' if converged else '否'
    print(f'是否收敛: {sig}')
    print(f'总耗时: {total_time:.1f}s')
    print(f'最终答案: {final_answer[:300]}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Config.OUTPUT_DIR / f'psoa_verifier_log_{timestamp}.md'

    log_content = '# PSOA + Verifier 完整日志\n\n'
    log_content += '- 时间: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n'
    log_content += '- 问题: ' + Config.QUESTION + '\n'
    log_content += '- 任务类型: ' + Config.TASK_TYPE + '\n'
    log_content += '- 最大轮数: ' + str(Config.MAX_ROUNDS) + '\n'
    log_content += '- 是否收敛: ' + str(converged) + '\n'
    log_content += '- 总耗时: ' + f'{total_time:.1f}s' + '\n'
    log_content += '\n## 最终答案\n' + final_answer + '\n'

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    print(f'\n[日志] 已保存: {log_path}')


if __name__ == '__main__':
    asyncio.run(main())
