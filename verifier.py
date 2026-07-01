#!/usr/bin/env python3
'''
PSOA External Verifier System
=============================
Hybrid Verifier layer that binds model outputs to verifiable reality constraints.

Architecture:
  Candidate Output -> SchemaVerifier -> ToolVerifier -> LLMJudge -> Aggregation
                                    v
                            PASS / REJECT / REVIEW

Designed for trajectory filtering -> SFT/DPO dataset generation.
'''

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# ===================================================================
# 1. Core Types
# ===================================================================

class Verdict(Enum):
    PASS = 'PASS'
    REJECT = 'REJECT'
    REVIEW = 'REVIEW'


@dataclass
class VerificationResult:
    layer: str
    passed: bool
    score: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateOutput:
    question: str
    candidate: str
    round_num: int
    strategy: str
    r_feedback: Optional[str] = None
    l_signal: Optional[int] = None
    raw_r_parsed: Optional[Dict] = None
    raw_l_parsed: Optional[Dict] = None


@dataclass
class VerifierReport:
    verdict: Verdict
    schema_result: VerificationResult
    tool_result: Optional[VerificationResult]
    llm_result: Optional[VerificationResult]
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    trajectory_data: Optional[Dict] = None


# ===================================================================
# 2. Schema Verifier - Deterministic structural checks
# ===================================================================

class SchemaVerifier:
    '''Validates structural integrity of candidate output.'''

    CONCLUSION_KEYWORDS = [
        '因此', '所以', '结论', '答案是', '最终', '综上',
        '综上所述', '故', '因而', '可见', '显然', '必然',
        'should', 'therefore', 'conclusion', 'answer is',
        '最终答案是',
    ]

    def __init__(self, min_length=20, require_conclusion=True):
        self.min_length = min_length
        self.require_conclusion = require_conclusion

    def verify(self, candidate):
        text = candidate.candidate.strip()
        issues = []
        scores = {}

        if len(text) < self.min_length:
            issues.append(f'输出过短 ({len(text)} chars < {self.min_length})')
        scores['length'] = min(1.0, len(text) / self.min_length)

        scores['json_valid'] = 1.0
        if text.startswith('{') or text.startswith('['):
            try:
                json.loads(text)
            except json.JSONDecodeError as e:
                issues.append(f'JSON格式错误: {e}')
                scores['json_valid'] = 0.0

        if self.require_conclusion:
            has_conclusion = any(kw in text for kw in self.CONCLUSION_KEYWORDS)
            scores['has_conclusion'] = 1.0 if has_conclusion else 0.3
            if not has_conclusion:
                issues.append('未检测到明确结论词')

        sentence_count = len(re.findall(r'[。！？；.!?;]', text))
        scores['coherence'] = min(1.0, sentence_count / 3.0)
        if sentence_count < 2:
            issues.append(f'文本过于简短，仅 {sentence_count} 个句子')

        overall = sum(scores.values()) / max(len(scores), 1)
        passed = overall >= 0.7 and len(issues) == 0

        return VerificationResult(
            layer='Schema',
            passed=passed,
            score=round(overall, 3),
            reason='; '.join(issues) if issues else '结构完整',
            details={'scores': scores, 'issues': issues},
        )


# ===================================================================
# 3. Tool Verifier - Reality-grounded checks
# ===================================================================

class ToolVerifier:
    '''Grounds candidate outputs in external reality.'''

    async def verify(self, candidate):
        text = candidate.candidate
        question = candidate.question
        results = []

        math_result = self._check_math(question, text)
        if math_result is not None:
            results.append(math_result)

        code_result = self._check_code(text)
        if code_result is not None:
            results.append(code_result)

        logic_result = self._check_logic(question, text)
        results.append(logic_result)

        if not results:
            return VerificationResult(
                layer='Tool', passed=True, score=1.0,
                reason='无可用工具检测', details={'fallback': True}
            )

        weights = []
        scored = []
        for r in results:
            if r.passed:
                weights.append(r.details.get('weight', 1.0))
                scored.append((r.score, r.reason))

        if not weights:
            overall_score = min(r.score for r in results)
            passed = False
        else:
            total_w = sum(weights)
            overall_score = sum(s * w for (s, _), w in zip(scored, weights)) / total_w
            passed = overall_score >= 0.6

        reasons = [r for _, r in scored] if scored else [r.reason for r in results]
        return VerificationResult(
            layer='Tool',
            passed=passed,
            score=round(overall_score, 3),
            reason='; '.join(reasons[:3]),
            details={'checks': [r.details for r in results]},
        )

    def _check_math(self, question, answer):
        matches = re.findall(r'(?:等于|=)\s*([\d\.]+)', answer)
        if not matches:
            return None
        num_in_a = re.findall(r'[\d\.]+', answer)
        try:
            answer_nums = [float(m) for m in num_in_a]
            if not answer_nums:
                return VerificationResult(
                    layer='Tool-Math', passed=True, score=1.0,
                    reason='无可验证数字', details={'weight': 0.5}
                )
            return VerificationResult(
                layer='Tool-Math', passed=True, score=0.8,
                reason=f'检测到数值答案 {answer_nums[0]}',
                details={'weight': 0.7, 'extracted': answer_nums}
            )
        except (ValueError, IndexError):
            return VerificationResult(
                layer='Tool-Math', passed=False, score=0.0,
                reason='数学验证解析失败', details={'weight': 0.7}
            )

    def _check_code(self, text):
        code_match = re.search(r'`(?:python|py)?\s*\n(.*?)`', text, re.DOTALL)
        if not code_match:
            return None
        code = code_match.group(1).strip()
        try:
            exec_globals = {}
            exec(code, exec_globals)
            return VerificationResult(
                layer='Tool-Code', passed=True, score=1.0,
                reason='代码执行成功', details={'weight': 1.0}
            )
        except Exception as e:
            return VerificationResult(
                layer='Tool-Code', passed=False, score=0.0,
                reason=f'代码执行失败: {e}', details={'weight': 1.0}
            )

    def _check_logic(self, question, answer):
        issues = []
        score = 1.0
        negations = ['不正确', '不是', '没有', '错误', '不对']
        affirmations = ['正确', '是的', '有', '对', '是']
        neg_count = sum(1 for w in negations if w in answer)
        aff_count = sum(1 for w in affirmations if w in answer)
        if neg_count >= 3 and aff_count >= 3:
            issues.append('可能存在自相矛盾表述')
            score -= 0.2
        step_keywords = ['第一步', '首先', '其次', '然后', '最后', '步骤1', '步骤2']
        if not any(kw in answer for kw in step_keywords):
            score -= 0.05
        q_kw = re.findall(r'[\u4e00-\u9fff]{2,}', question)
        a_kw = re.findall(r'[\u4e00-\u9fff]{2,}', answer)
        if q_kw and a_kw:
            common = set(q_kw) & set(a_kw)
            alignment = len(common) / max(len(q_kw), 1)
            score *= alignment
            if alignment < 0.1:
                issues.append('答案与问题关键词重合度低')
        passed = score >= 0.6 and len([i for i in issues if '矛盾' in i]) == 0
        return VerificationResult(
            layer='Tool-Logic',
            passed=passed,
            score=round(max(0.0, score), 3),
            reason='; '.join(issues) if issues else '逻辑自洽',
            details={'weight': 0.6, 'alignment': round(score, 3)},
        )


# ===================================================================
# 4. LLM Judge - Semantic quality scoring
# ===================================================================

class LLMJudge:
    '''Uses LLM to score semantic quality. Never the sole decider.'''

    SYSTEM_PROMPT = '''你是一个严格的外部裁判（Verifier）。你的职责是评估候选答案的质量，
而不是修改它。请基于以下维度打分（0.0-1.0）：

1. 准确性 (accuracy): 答案是否与事实/逻辑一致
2. 完整性 (completeness): 是否覆盖了问题的所有方面
3. 推理质量 (reasoning): 推理过程是否清晰、有依据
4. 相关性 (relevance): 答案是否直接回应了问题

请严格按以下JSON格式输出：
{
  "accuracy": 0.0-1.0,
  "completeness": 0.0-1.0,
  "reasoning_quality": 0.0-1.0,
  "relevance": 0.0-1.0,
  "weighted_score": 0.0-1.0,
  "verdict_hint": "PASS|REVIEW|REJECT",
  "reason": "简要评语（中文）"
}

加权公式：weighted = accuracy*0.4 + completeness*0.2 + reasoning_quality*0.25 + relevance*0.15
'''

    def __init__(self, client=None):
        self.client = client

    async def verify(self, candidate):
        if self.client is None:
            return VerificationResult(
                layer='LLM-Judge', passed=True, score=1.0,
                reason='LLM客户端未配置，跳过语义评分',
                details={'skipped': True}
            )
        r_fb = candidate.r_feedback if candidate.r_feedback else '无'
        l_sig = '放行 (1)' if candidate.l_signal == 1 else ('不放行 (0)' if candidate.l_signal == 0 else '未知')
        user_prompt = f'''【问题】
{candidate.question}

【候选答案】（第{candidate.round_num}轮）
{candidate.candidate}

【R线程反馈】
{r_fb}

【L线程信号】
{l_sig}

请评分。'''
        try:
            response = await self.client.chat.completions.create(
                model='agnes-2.0-flash',
                messages=[
                    {'role': 'system', 'content': self.SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = {'error': 'no json found'}

            accuracy = parsed.get('accuracy', 0.5)
            completeness = parsed.get('completeness', 0.5)
            reasoning = parsed.get('reasoning_quality', 0.5)
            relevance = parsed.get('relevance', 0.5)
            weighted = parsed.get('weighted_score',
                accuracy * 0.4 + completeness * 0.2 + reasoning * 0.25 + relevance * 0.15)
            weighted = max(0.0, min(1.0, weighted))

            if weighted > 0.75:
                passed = True
            elif weighted > 0.5:
                passed = True
            else:
                passed = False

            return VerificationResult(
                layer='LLM-Judge',
                passed=passed,
                score=round(weighted, 3),
                reason=parsed.get('reason', '无评语'),
                details={
                    'accuracy': accuracy,
                    'completeness': completeness,
                    'reasoning_quality': reasoning,
                    'relevance': relevance,
                    'weighted_score': round(weighted, 3),
                    'verdict_hint': parsed.get('verdict_hint', 'UNKNOWN'),
                },
            )
        except Exception as e:
            return VerificationResult(
                layer='LLM-Judge',
                passed=True,
                score=0.5,
                reason=f'LLM评分异常: {e}',
                details={'error': str(e), 'fallback': True},
            )


# ===================================================================
# 5. Aggregation Layer - Final verdict
# ===================================================================

class AggregationLayer:
    def __init__(
        self,
        schema_threshold=0.7,
        tool_threshold=0.6,
        llm_pass_threshold=0.75,
        llm_review_threshold=0.5,
        tool_strict=True,
    ):
        self.schema_threshold = schema_threshold
        self.tool_threshold = tool_threshold
        self.llm_pass_threshold = llm_pass_threshold
        self.llm_review_threshold = llm_review_threshold
        self.tool_strict = tool_strict

    def aggregate(self, schema, tool, llm):
        reasons = []
        scores = {}
        scores['schema'] = schema.score
        if tool:
            scores['tool'] = tool.score
        if llm:
            scores['llm'] = llm.score

        if schema.score < self.schema_threshold:
            reasons.append(f'Schema校验失败 (score={schema.score:.2f}): {schema.reason}')
            return VerifierReport(
                verdict=Verdict.REJECT,
                schema_result=schema,
                tool_result=tool,
                llm_result=llm,
                scores=scores,
                reasons=reasons,
            )

        if tool and tool.score < self.tool_threshold:
            if self.tool_strict and tool.layer in ('Tool-Math', 'Tool-Code'):
                reasons.append(f'Tool校验失败 (score={tool.score:.2f}): {tool.reason}')
                return VerifierReport(
                    verdict=Verdict.REJECT,
                    schema_result=schema,
                    tool_result=tool,
                    llm_result=llm,
                    scores=scores,
                    reasons=reasons,
                )
            reasons.append(f'Tool校验较弱 (score={tool.score:.2f}): {tool.reason}')

        if llm:
            if llm.score >= self.llm_pass_threshold:
                verdict = Verdict.PASS
                reasons.append(f'LLM语义评分通过 (score={llm.score:.2f})')
            elif llm.score >= self.llm_review_threshold:
                verdict = Verdict.REVIEW
                reasons.append(f'LLM语义评分需复审 (score={llm.score:.2f})')
            else:
                verdict = Verdict.REJECT
                reasons.append(f'LLM语义评分过低 (score={llm.score:.2f})')
        else:
            if tool and tool.score >= self.tool_threshold:
                verdict = Verdict.PASS
                reasons.append('Schema+Tool双通过，无LLM降级判定')
            else:
                verdict = Verdict.REVIEW
                reasons.append('无LLM评分，保守判定为REVIEW')

        return VerifierReport(
            verdict=verdict,
            schema_result=schema,
            tool_result=tool,
            llm_result=llm,
            scores=scores,
            reasons=reasons,
        )


# ===================================================================
# 6. Main Verifier - Orchestrator
# ===================================================================

class Verifier:
    '''Hybrid Verifier: main entry point.'''

    def __init__(
        self,
        client=None,
        schema_threshold=0.7,
        tool_threshold=0.6,
        llm_pass_threshold=0.75,
        llm_review_threshold=0.5,
        tool_strict=True,
        enable_llm_judge=True,
    ):
        self.client = client
        self.enable_llm_judge = enable_llm_judge
        self.schema_verifier = SchemaVerifier()
        self.tool_verifier = ToolVerifier()
        self.llm_judge = LLMJudge(client=client if enable_llm_judge else None)
        self.aggregator = AggregationLayer(
            schema_threshold=schema_threshold,
            tool_threshold=tool_threshold,
            llm_pass_threshold=llm_pass_threshold,
            llm_review_threshold=llm_review_threshold,
            tool_strict=tool_strict,
        )

    async def verify(
        self,
        question,
        candidate,
        round_num=1,
        strategy='',
        r_feedback=None,
        l_signal=None,
        r_parsed=None,
        l_parsed=None,
    ):
        start = time.time()
        c = CandidateOutput(
            question=question,
            candidate=candidate,
            round_num=round_num,
            strategy=strategy,
            r_feedback=r_feedback,
            l_signal=l_signal,
            raw_r_parsed=r_parsed,
            raw_l_parsed=l_parsed,
        )
        schema_result = self.schema_verifier.verify(c)
        tool_result = await self.tool_verifier.verify(c)
        llm_result = None
        if self.enable_llm_judge:
            llm_result = await self.llm_judge.verify(c)
        report = self.aggregator.aggregate(schema_result, tool_result, llm_result)
        report.elapsed_ms = (time.time() - start) * 1000
        report.trajectory_data = {
            'question': question,
            'candidate': candidate,
            'round_num': round_num,
            'verdict': report.verdict.value,
            'scores': report.scores,
            'reasons': report.reasons,
            'strategy_at_round': strategy,
            'r_feedback_summary': r_feedback[:500] if r_feedback else None,
            'l_signal': l_signal,
            'elapsed_ms': report.elapsed_ms,
        }
        return report

    def format_report(self, report):
        lines = [
            '=' * 50,
            f'VERIFIER REPORT | Verdict: **{report.verdict.value}**',
            '=' * 50,
            '',
        ]
        if 'schema' in report.scores:
            lines.append(f"Schema  (score={report.scores['schema']}): {report.schema_result.reason}")
        if report.tool_result:
            lines.append(f"Tool    (score={report.scores.get('tool', '?')}): {report.tool_result.reason}")
        if report.llm_result:
            lines.append(f"LLM     (score={report.scores.get('llm', '?')}): {report.llm_result.reason}")
        lines.extend([''])
        for r in report.reasons:
            lines.append(f'  -> {r}')
        lines.extend([
            '',
            f'Elapsed: {report.elapsed_ms:.0f}ms',
            '=' * 50,
        ])
        return chr(10).join(lines)


# ===================================================================
# 7. Trajectory Exporter - For SFT/DPO dataset generation
# ===================================================================

class TrajectoryExporter:
    '''Exports verified trajectories into training-ready formats.'''

    def __init__(self, output_dir='./verifier_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.passed = []
        self.rejected = []
        self.reviewed = []

    def ingest(self, report, candidate):
        data = report.trajectory_data
        if data is None:
            return
        entry = {
            'question': candidate.question,
            'candidate': candidate.candidate,
            'round': candidate.round_num,
            'strategy': candidate.strategy,
            'verdict': report.verdict.value,
            'scores': report.scores,
            'reasons': report.reasons,
            'elapsed_ms': report.elapsed_ms,
        }
        if report.verdict == Verdict.PASS:
            self.passed.append(entry)
        elif report.verdict == Verdict.REJECT:
            self.rejected.append(entry)
        else:
            self.reviewed.append(entry)

    def export_sft_jsonl(self, filepath=None):
        fp = filepath or str(self.output_dir / 'sft_trajectories.jsonl')
        with open(fp, 'w', encoding='utf-8') as f:
            for entry in self.passed:
                record = {
                    'messages': [
                        {'role': 'user', 'content': entry['question']},
                        {'role': 'assistant', 'content': entry['candidate']},
                    ],
                    'metadata': {
                        'verdict': entry['verdict'],
                        'scores': entry['scores'],
                        'round': entry['round'],
                    }
                }
                f.write(json.dumps(record, ensure_ascii=False) + chr(10))
        return fp

    def export_dpo_jsonl(self, filepath=None):
        fp = filepath or str(self.output_dir / 'dpo_trajectories.jsonl')
        question_map = {}
        for entry in self.passed + self.rejected:
            q = entry['question']
            if q not in question_map:
                question_map[q] = {'passed': [], 'rejected': []}
            if entry['verdict'] == 'PASS':
                question_map[q]['passed'].append(entry)
            else:
                question_map[q]['rejected'].append(entry)
        with open(fp, 'w', encoding='utf-8') as f:
            for question, pairs in question_map.items():
                for chosen in pairs['passed']:
                    for rejected in pairs['rejected']:
                        record = {
                            'prompt': question,
                            'chosen': {
                                'content': chosen['candidate'],
                                'scores': chosen['scores'],
                            },
                            'rejected': {
                                'content': rejected['candidate'],
                                'scores': rejected['scores'],
                            },
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + chr(10))
        return fp

    def export_summary(self, filepath=None):
        fp = filepath or str(self.output_dir / 'verifier_summary.json')
        total = len(self.passed) + len(self.rejected) + len(self.reviewed)
        summary = {
            'total': total,
            'passed': len(self.passed),
            'rejected': len(self.rejected),
            'reviewed': len(self.reviewed),
            'pass_rate': round(len(self.passed) / max(total, 1), 3),
            'avg_scores': {
                k: round(sum(e['scores'].get(k, 0) for e in self.passed) / max(len(self.passed), 1), 3)
                for k in ['schema', 'tool', 'llm']
            },
            'by_round': {},
        }
        for entry in self.passed + self.rejected + self.reviewed:
            r = str(entry['round'])
            if r not in summary['by_round']:
                summary['by_round'][r] = {'pass': 0, 'reject': 0, 'review': 0}
            verdict = entry['verdict']
            if verdict == 'PASS':
                summary['by_round'][r]['pass'] += 1
            elif verdict == 'REJECT':
                summary['by_round'][r]['reject'] += 1
            else:
                summary['by_round'][r]['review'] += 1
        with open(fp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return fp


# ===================================================================
# 8. PSOA Integration Helper
# ===================================================================

def build_candidate_from_psoa(
    question,
    candidate,
    round_num,
    strategy,
    r_parsed=None,
    l_parsed=None,
):
    r_feedback = None
    if r_parsed:
        errors = r_parsed.get('errors', [])
        overall = r_parsed.get('overall_assessment', '')
        r_feedback = json.dumps({'errors': errors, 'summary': overall}, ensure_ascii=False)
    l_signal = None
    if l_parsed:
        l_signal = l_parsed.get('let_through', None)
    return CandidateOutput(
        question=question,
        candidate=candidate,
        round_num=round_num,
        strategy=strategy,
        r_feedback=r_feedback,
        l_signal=l_signal,
        raw_r_parsed=r_parsed,
        raw_l_parsed=l_parsed,
    )


# ===================================================================
# 9. Demo / Test
# ===================================================================

async def _demo():
    '''Quick demo of the Verifier system.'''
    print('=' * 60)
    print('PSOA Verifier - Demo')
    print('=' * 60)

    question = '如果所有的猫都是动物，所有的动物都有生命，那么所有的猫都有生命吗？'
    good_candidate = '''这是一个经典的三段论推理问题。

首先，我们明确前提条件：
1. 所有的猫都是动物
2. 所有的动物都有生命

接下来进行逻辑推理：
- 既然所有的猫都属于动物的范畴（前提1）
- 而所有的动物都有生命（前提2）
- 那么根据传递性原理，所有的猫也都有生命

结论：是的，所有的猫都有生命。这是演绎推理中典型的三段论形式，结论必然成立。'''

    bad_candidate = '嗯，应该是吧。猫是动物，动物有东西，所以猫有东西。'

    print()
    print('--- Test 1: Good candidate (Schema + Tool only) ---')
    verifier1 = Verifier(enable_llm_judge=False)
    report1 = await verifier1.verify(
        question=question,
        candidate=good_candidate,
        round_num=1,
    )
    print(verifier1.format_report(report1))
    print(f'Verdict: {report1.verdict.value}')

    print()
    print('--- Test 2: Bad candidate (Schema + Tool only) ---')
    verifier2 = Verifier(enable_llm_judge=False)
    report2 = await verifier2.verify(
        question=question,
        candidate=bad_candidate,
        round_num=2,
    )
    print(verifier2.format_report(report2))
    print(f'Verdict: {report2.verdict.value}')

    print()
    print('--- Test 3: Trajectory Export ---')
    exporter = TrajectoryExporter('./verifier_output')

    class FakeCandidate:
        def __init__(self, q, c, r, s=''):
            self.question = q
            self.candidate = c
            self.round_num = r
            self.strategy = s
            self.r_feedback = None
            self.l_signal = None
            self.raw_r_parsed = None
            self.raw_l_parsed = None

    exporter.ingest(report1, FakeCandidate(question, good_candidate, 1))
    exporter.ingest(report2, FakeCandidate(question, bad_candidate, 2))

    sft_path = exporter.export_sft_jsonl()
    dpo_path = exporter.export_dpo_jsonl()
    summary_path = exporter.export_summary()

    print(f'SFT trajectories: {sft_path}')
    print(f'DPO trajectories: {dpo_path}')
    print(f'Summary: {summary_path}')

    print()
    print('=' * 60)
    print('Demo complete!')
    print('=' * 60)


if __name__ == '__main__':
    asyncio.run(_demo())