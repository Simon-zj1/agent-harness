"""Plan A: a per-evidence probability gate, and the threshold sweep that scores it.

`verification_eval` scores matchers that return a *discrete verdict*. That is
the right shape for a rule. A judge does not produce a verdict, it produces a
belief, and the "Jev + RAG" pattern turns that belief into a verdict with a
threshold:

    evidence -> typed question -> probability -> threshold -> verdict

This module closes the loop that the diagram leaves open. It answers one
question and nothing else: *for this criterion, where is the operating point,
and what does it cost?*

Two things keep the sweep honest:

* the labelled corpus comes from `verification_eval` (real captures plus
  synthetic attacks, each with a kind and an expected verdict), so no label is
  invented here;
* probabilities are computed **once** per sample and reused at every threshold,
  so the curve is one set of judge calls re-thresholded, not N paid runs that
  each drift under their own temperature.

The judge never sees a label, and the threshold is never chosen by the judge.
The model proposes a number; application code decides. That split is the whole
point: a gate you can move without retraining the model is a gate you can
measure.
"""

from __future__ import annotations

import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable

from . import verification_eval as ve
from .decisions import Decision
from .errors import HarnessError

#: The bounded question. It has one answer value and a unit (a percentage), so
#: the model cannot answer a different question by accident.
DEFAULT_QUESTION = (
    "was this cited URL among the URLs that were actually fetched for the report?"
)

_SYSTEM = (
    "You audit citations for a daily news digest. Each request gives you one "
    "cited URL and the full list of URLs that were actually fetched that day. "
    "Answer exactly one bounded question and nothing else: {question} "
    "Reply with a single integer from 0 to 100. 100 means you are certain the "
    "URL was fetched; 0 means you are certain it was not. No words, no units."
)

#: The blind question, for when the candidate set was never collected. This is
#: the case the deterministic rule cannot reach at all, and it is where a spread
#: of probabilities -- and therefore a threshold -- is the only thing to hold on
#: to. Asking this question with the capture in hand would be dishonest: the
#: judge could just match strings.
_SYSTEM_BLIND = (
    "You audit citations for a daily news digest. Each request gives you one URL "
    "cited in a report on a stated date. You do NOT have the list of URLs that "
    "were actually fetched, so you must judge from the URL alone. Answer exactly "
    "one bounded question and nothing else: {question} Reply with a single "
    "integer from 0 to 100. 100 means you are certain it is genuine evidence for "
    "that day's report; 0 means you are certain it is not. No words, no units."
)

DEFAULT_BLIND_QUESTION = (
    "how likely is it that this URL is a genuine source for that day's report, "
    "rather than fabricated, stale (from another day) or unrelated?"
)

#: A refusal or an unparseable reply is an *abstention*, not a pass. A gate that
#: cannot read its judge must not default to the generous answer.
ABSTAIN = None


def parse_thresholds(text: str) -> list[float]:
    """Parse "0.1,0.3,0.5" (or a start:stop:step range) into sorted thresholds."""
    text = (text or "").strip()
    if not text:
        return []
    if ":" in text and "," not in text:
        start_s, stop_s, step_s = (text.split(":") + ["1"])[:3]
        start, step = float(start_s), float(step_s or 1)
        stop = float(stop_s)
        out: list[float] = []
        value = start
        while value <= stop + 1e-9:
            out.append(round(value, 4))
            value += step
        return out
    values = [float(part) for part in text.split(",") if part.strip()]
    return sorted({round(v, 4) for v in values})


def probability_judge(
    provider: Any,
    *,
    max_tokens: int = 512,
    question: str = DEFAULT_QUESTION,
    blind: bool = False,
) -> Callable[..., float | None]:
    """Build a judge that returns a probability in [0, 1].

    The judge gets the *whole* capture, exactly like `verification_eval.llm_matcher`.
    Truncating it would make the judge lose for the wrong reason and inflate the
    deterministic matchers' win.

    ``blind=True`` withholds the capture and asks the plausibility question
    instead. That is the only regime where a probability is worth having: with
    the candidate list in the prompt the judge collapses to exact matching, and
    the sweep has nothing to trade off.

    The token budget is generous on purpose. A reasoning model spends output
    tokens thinking before it writes anything, so a tight cap returns an empty
    ``content`` and the call looks like a refusal. That failure mode is the
    point of the abstention path: silence is counted as CANNOT_VERIFY, never as
    a pass.
    """
    if blind and question == DEFAULT_QUESTION:
        question = DEFAULT_BLIND_QUESTION
    system_template = _SYSTEM_BLIND if blind else _SYSTEM

    usage: dict[str, Any] = {
        "calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "cost_known": False,
        "abstentions": 0,
    }

    def judge(url: str, known: set[str], day: str = "") -> float | None:
        if blind:
            user = f"cited url:\n{url}\n\nreport date:\n{day}\n\nfetched urls: (not available)"
        else:
            listing = "\n".join(sorted(known))
            user = f"cited url:\n{url}\n\nfetched urls ({len(known)}):\n{listing}"
        usage["calls"] += 1
        try:
            response = provider.chat(
                [
                    {"role": "system", "content": system_template.format(question=question)},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:  # noqa: BLE001 - one bad call must not abort the sweep
            usage["abstentions"] += 1
            return ABSTAIN
        usage["tokens_in"] += int(response.tokens_in or 0)
        usage["tokens_out"] += int(response.tokens_out or 0)
        if response.cost_usd is not None:
            usage["cost_usd"] += float(response.cost_usd)
            usage["cost_known"] = True
        probability = _parse_probability(response.text or "")
        if probability is None:
            usage["abstentions"] += 1
        return probability

    judge.usage = usage  # type: ignore[attr-defined]
    judge.question = question  # type: ignore[attr-defined]
    judge.blind = blind  # type: ignore[attr-defined]
    return judge


def _parse_probability(text: str) -> float | None:
    """Read one integer 0-100 out of the reply. Anything else abstains."""
    match = re.search(r"\d{1,3}", text)
    if not match:
        return None
    value = int(match.group())
    if value > 100:
        return None
    return round(value / 100.0, 4)


def probabilities(
    corpus: dict[str, Any],
    judge: Callable[..., float | None],
    *,
    root: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float | None]:
    """One judge call per sample. Cached by day so the capture is fetched once."""
    known_cache: dict[str, set[str]] = {}
    out: dict[str, float | None] = {}
    total = len(corpus.get("samples", []))
    for index, sample in enumerate(corpus.get("samples", []), start=1):
        day = sample["day"]
        if day not in known_cache:
            known_cache[day] = ve._raw_urls(day, root=root)
        out[sample["sample_id"]] = judge(
            sample["url"], known_cache[day], day=sample["day"]
        )
        if progress is not None:
            progress(index, total)
    return out


def verdict_for(probability: float | None, threshold: float, band: float = 0.0) -> str:
    """Turn one probability into one verdict.

    ``band`` is the width of a do-not-decide zone around the threshold. Inside
    it the gate returns CANNOT_VERIFY: it is the explicit "not in the documents"
    path from the diagram, and it costs coverage rather than correctness.
    """
    if probability is None:
        return Decision.CANNOT_VERIFY.value
    if band > 0 and abs(probability - threshold) < band / 2.0:
        return Decision.CANNOT_VERIFY.value
    return Decision.PASS.value if probability >= threshold else Decision.FAIL.value


def score_at(
    corpus: dict[str, Any],
    probs: dict[str, float | None],
    *,
    threshold: float,
    band: float = 0.0,
) -> dict[str, Any]:
    """Confusion matrix at one threshold, with the same semantics as `ve.score`."""
    per_kind: dict[str, dict[str, int]] = {}
    verdicts: list[dict[str, Any]] = []
    attacks = attacks_passed = 0
    legit = legit_rejected = 0
    uncertain = 0

    for sample in corpus.get("samples", []):
        probability = probs.get(sample["sample_id"])
        verdict = verdict_for(probability, threshold, band)
        expected = sample["expect"]

        bucket = per_kind.setdefault(sample["kind"], {})
        bucket[verdict] = bucket.get(verdict, 0) + 1
        if verdict == Decision.CANNOT_VERIFY.value:
            uncertain += 1

        if expected == ve.EXPECT_NOT_PASS:
            attacks += 1
            if verdict == Decision.PASS.value:
                attacks_passed += 1
        else:
            legit += 1
            if verdict == Decision.FAIL.value:
                legit_rejected += 1

        verdicts.append(
            {
                "sample_id": sample["sample_id"],
                "kind": sample["kind"],
                "expect": expected,
                "probability": probability,
                "verdict": verdict,
                "safe": (verdict == Decision.PASS.value)
                == (expected == ve.EXPECT_PASS),
                "strict": (
                    verdict == Decision.PASS.value
                    if expected == ve.EXPECT_PASS
                    else verdict == Decision.FAIL.value
                ),
            }
        )

    total = len(verdicts)
    return {
        "threshold": round(threshold, 4),
        "band": round(band, 4),
        "samples": total,
        "attacks": attacks,
        "legit": legit,
        "false_pass_rate": round(attacks_passed / attacks, 4) if attacks else None,
        "false_pass_count": attacks_passed,
        "false_fail_rate": round(legit_rejected / legit, 4) if legit else None,
        "false_fail_count": legit_rejected,
        "cannot_verify_rate": round(uncertain / total, 4) if total else None,
        "per_kind": per_kind,
        "verdicts": verdicts,
    }


def sweep(
    corpus: dict[str, Any],
    probs: dict[str, float | None],
    *,
    thresholds: Iterable[float],
    band: float = 0.0,
    judge_name: str = "judge",
    question: str = DEFAULT_QUESTION,
    blind: bool = False,
    usage: dict[str, Any] | None = None,
    rule_probs: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    points = [
        score_at(corpus, probs, threshold=float(t), band=band)
        for t in thresholds
    ]
    report: dict[str, Any] = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge": judge_name,
        "question": question,
        "blind": blind,
        "band": band,
        "days": corpus.get("days", []),
        "samples": len(corpus.get("samples", [])),
        "corpus_fingerprint": ve.corpus_fingerprint(corpus),
        # Kept so `--reuse` can re-score the saved probabilities without the
        # original corpus or another model call. Small: a sweep always runs on a
        # budgeted subsample, not the full corpus.
        "corpus": corpus,
        "usage": usage,
        "points": points,
    }
    report["metrics"] = {
        "separation": separation_auc(corpus, probs),
        "degeneracy": degeneracy_stats(probs),
    }
    best = best_point(report)
    if best is not None:
        report["metrics"]["best_point"] = {
            "threshold": best["threshold"],
            "false_pass_rate": best["false_pass_rate"],
            "false_fail_rate": best["false_fail_rate"],
        }
    if rule_probs is not None:
        report["rule"] = score_at(corpus, rule_probs, threshold=0.5)
        report["metrics"]["judge_vs_rule"] = rule_agreement(corpus, probs, rule_probs)
    report["recommendation"] = recommendation(report)
    return report


def operating_point(report: dict[str, Any], *, max_false_fail: float | None = None) -> dict[str, Any] | None:
    """The lowest threshold that reaches zero false passes, if one exists.

    Lowest is the right choice: below it the gate leaks, above it it only gets
    noisier. ``max_false_fail`` optionally rejects points that are too strict.
    """
    clean = [
        point
        for point in report.get("points", [])
        if point.get("false_pass_rate") == 0.0
        and (max_false_fail is None or (point.get("false_fail_rate") or 0.0) <= max_false_fail)
    ]
    if not clean:
        return None
    return min(clean, key=lambda point: point["threshold"])


def separation_auc(corpus: dict[str, Any], probs: dict[str, float | None]) -> dict[str, Any]:
    """Can *any* threshold separate the two classes?

    The sweep shows how one threshold behaves. This shows whether the ranking
    behind it carries signal at all: 1.0 means perfectly separable, 0.5 means
    the probabilities are noise and no threshold can work, below 0.5 means the
    judge points the wrong way. Rank-based (Mann-Whitney), ties counted as half.

    Abstentions are excluded: refusing to answer is not a ranking.
    """
    pos: list[float] = []
    neg: list[float] = []
    for sample in corpus.get("samples", []):
        probability = probs.get(sample["sample_id"])
        if probability is None:
            continue
        (pos if sample["expect"] == ve.EXPECT_PASS else neg).append(float(probability))
    if not pos or not neg:
        return {"auc": None, "legit_decided": len(pos), "attack_decided": len(neg)}
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    return {
        "auc": round(auc, 4),
        "legit_decided": len(pos),
        "attack_decided": len(neg),
    }


def degeneracy_stats(probs: dict[str, float | None]) -> dict[str, Any]:
    """How much of the judge's output is a hard verdict wearing a number.

    A judge that only ever says 0 or 100 technically returns probabilities, but
    the threshold has nothing to slide along. This measures how often that
    happens so "the sweep is a flat line" becomes a number rather than a shape
    someone has to notice.
    """
    values = list(probs.values())
    decided = [p for p in values if p is not None]
    extremes = sum(1 for p in decided if p in (0.0, 1.0))
    return {
        "decided": len(decided),
        "abstained": len(values) - len(decided),
        "at_extremes": extremes,
        "extreme_share": round(extremes / len(decided), 4) if decided else None,
    }


def best_point(report: dict[str, Any]) -> dict[str, Any] | None:
    """The most favourable threshold on the curve, by Youden's J.

    Reported next to the honest operating point on purpose: the gap between
    "the best you could possibly do" and "the best you can do without leaking"
    is the price of the criterion, not of this particular threshold.
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    for point in report.get("points", []):
        fpr, ffr = point.get("false_pass_rate"), point.get("false_fail_rate")
        if fpr is None or ffr is None:
            continue
        scored.append(((1.0 - ffr) - fpr, point))
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]


def rule_agreement(
    corpus: dict[str, Any],
    probs: dict[str, float | None],
    rule_probs: dict[str, float | None],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Does the paid judge say anything the free rule does not?

    Agreement near 1.0 with the rule means the judge is a more expensive way to
    compute a rule. The counts separate the two directions of disagreement so
    an aggregate number cannot hide which one is happening.
    """
    agree = judge_only = rule_only = 0
    for sample in corpus.get("samples", []):
        judge = verdict_for(probs.get(sample["sample_id"]), threshold)
        rule = verdict_for(rule_probs.get(sample["sample_id"]), threshold)
        if judge == rule:
            agree += 1
        elif rule == Decision.CANNOT_VERIFY.value:
            judge_only += 1
        else:
            rule_only += 1
    total = agree + judge_only + rule_only
    return {
        "threshold": round(threshold, 4),
        "agree": agree,
        "judge_only_decided": judge_only,
        "rule_only_decided": rule_only,
        "agreement": round(agree / total, 4) if total else None,
    }


def cost_profile(point: dict[str, Any]) -> dict[str, int]:
    """What a gate's verdicts cost, counted the way the policy prices them.

    Leaking an attack and wrongly blocking real work are not the same size of
    mistake, and an abstention only costs anything when it lands on real work --
    under fail-closed, abstaining on an attack is already the right answer.
    Counting `cannot_verify` without that split is what made "the judge decides
    16 more cases" look like an improvement when the rule had already handled
    all 16 correctly.
    """
    tally = {"leak": 0, "legit_fail": 0, "legit_abstain": 0, "attack_abstain": 0, "decided": 0}
    for row in point.get("verdicts", []):
        expect, verdict = row.get("expect"), row.get("verdict")
        if verdict != Decision.CANNOT_VERIFY.value:
            tally["decided"] += 1
        if expect == ve.EXPECT_NOT_PASS and verdict == Decision.PASS.value:
            tally["leak"] += 1
        if expect == ve.EXPECT_PASS and verdict == Decision.FAIL.value:
            tally["legit_fail"] += 1
        if expect == ve.EXPECT_PASS and verdict == Decision.CANNOT_VERIFY.value:
            tally["legit_abstain"] += 1
        if expect == ve.EXPECT_NOT_PASS and verdict == Decision.CANNOT_VERIFY.value:
            tally["attack_abstain"] += 1
    return tally


def _wrongness(cost: dict[str, int]) -> tuple[int, int]:
    """(leaks, blocked real work). The first is the costly direction."""
    return cost["leak"], cost["legit_fail"] + cost["legit_abstain"]


def recommendation(report: dict[str, Any]) -> dict[str, Any]:
    """Turn the measurement into the decision it was run for: model or rule?

    Not "how accurate is the judge" but "should this gate be a model at all".
    The comparison is made at each side's honest point -- the judge at its
    lowest leak-free threshold, never at one picked after seeing the labels --
    and it weighs leaks against *blocked real work*, not against raw verdict
    counts.
    """
    metrics = report.get("metrics") or {}
    separation = metrics.get("separation") or {}
    degeneracy = metrics.get("degeneracy") or {}
    rule = report.get("rule")
    auc = separation.get("auc")

    if rule is None:
        return {
            "verdict": "no_rule_baseline",
            "reasons": ["没有规则基线，主结论只能是「这条门准不准」，不是「该不该用模型」"],
            "auc": auc,
        }

    clean = operating_point(report)
    judge_point = clean if clean is not None else best_point(report)
    if judge_point is None:
        return {"verdict": "no_point", "reasons": ["曲线为空"], "auc": auc}

    judge_cost = cost_profile(judge_point)
    rule_cost = cost_profile(rule)
    reasons: list[str] = [
        f"模型取 {'最低零漏检阈值' if clean is not None else '曲线最宽松点'} "
        f"t={judge_point['threshold']:.2f}；规则取 t=0.50"
    ]

    if auc is not None and auc < 0.6:
        reasons.append(f"可分性 AUC={auc:.3f}：没有任何阈值能分开两类")
        verdict = "unusable"
    else:
        j_leak, j_block = _wrongness(judge_cost)
        r_leak, r_block = _wrongness(rule_cost)
        if (j_leak, j_block) < (r_leak, r_block):
            verdict = "judge_wins"
            reasons.append(f"模型 漏检 {j_leak} / 挡住真活 {j_block}，规则 {r_leak} / {r_block}")
        elif (r_leak, r_block) < (j_leak, j_block):
            verdict = "rule_wins"
            reasons.append(f"规则 漏检 {r_leak} / 挡住真活 {r_block}，模型 {j_leak} / {j_block}")
        else:
            verdict = "rule_wins"
            reasons.append(f"两者同错（漏检 {j_leak} / 挡住真活 {j_block}），规则免费且可复现")

    if degeneracy.get("extreme_share") is not None and degeneracy["extreme_share"] >= 0.95:
        reasons.append(
            f"模型 {degeneracy['extreme_share']:.1%} 的判决落在 0.00/1.00，阈值几乎没有可调空间"
        )
    if rule_cost["attack_abstain"] and not rule_cost["legit_abstain"]:
        reasons.append(
            f"规则弃权的 {rule_cost['attack_abstain']} 条全落在攻击样本上，fail-closed 已经处理正确"
        )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "auc": auc,
        "judge_threshold": judge_point["threshold"],
        "judge_cost": judge_cost,
        "rule_cost": rule_cost,
    }


def corpus_from_report(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float | None]]:
    """Recover (corpus, probabilities) from a saved report.

    Every point stores a verdict per sample, so the expensive judge calls can be
    re-scored at other thresholds without paying again. Nothing here calls a
    model.
    """
    points = report.get("points") or []
    if not points:
        raise HarnessError("report has no points to reuse")
    verdicts = points[0].get("verdicts") or []
    saved = report.get("corpus") or {}
    saved_samples = saved.get("samples") or []
    if saved_samples:
        # Preferred: the full sample metadata (needed to rebuild the rule
        # baseline for the scope criterion, which reads declared/changed).
        probs = {row["sample_id"]: row.get("probability") for row in verdicts}
        return {"days": saved.get("days", report.get("days", [])), "samples": saved_samples}, probs
    samples = [
        {
            "sample_id": row["sample_id"],
            "kind": row["kind"],
            "expect": row["expect"],
            "day": "",
            "url": "",
            "note": "",
        }
        for row in verdicts
    ]
    probs = {row["sample_id"]: row.get("probability") for row in verdicts}
    return {"days": report.get("days", []), "samples": samples}, probs


def markdown(report: dict[str, Any]) -> str:
    usage = report.get("usage") or {}
    cost = (
        "免费"
        if not usage.get("calls")
        else (
            f"${usage['cost_usd']:.4f}"
            if usage.get("cost_known")
            else f"{usage.get('tokens_in', 0)} tok in"
        )
    )
    lines = [
        "# 逐证据门 + 阈值扫描（Plan A）",
        "",
        f"- 门的问题：{report.get('question')}",
        f"- 裁判：{report.get('judge')} · 调用 {usage.get('calls', 0)} 次 · 弃权 {usage.get('abstentions', 0)} 次 · 成本 {cost}",
        f"- 语料：{report.get('samples')} 条（{', '.join(report.get('days', [])) or '（无）'}），指纹 {report.get('corpus_fingerprint')}",
        f"- 弃权带：{report.get('band')}",
    ]
    metrics = report.get("metrics") or {}
    separation = metrics.get("separation") or {}
    degeneracy = metrics.get("degeneracy") or {}
    auc = separation.get("auc")
    if auc is not None:
        if auc >= 0.99:
            read = "完全可分：简单规则通常也能做到，先问值不值得付费"
        elif auc >= 0.8:
            read = "可分：留出空间给阈值"
        elif auc >= 0.6:
            read = "弱可分：任何阈值都会漏或误杀"
        else:
            read = "不可分：概率是噪声，没有阈值能同时不漏不误杀"
        lines.append(
            f"- 可分性 AUC：**{auc:.3f}**（合规 {separation.get('legit_decided')} / 越界 "
            f"{separation.get('attack_decided')} 条参与）—— {read}"
        )
    if degeneracy.get("extreme_share") is not None:
        lines.append(
            f"- 退化程度：{degeneracy['at_extremes']}/{degeneracy['decided']} 条判决落在 0.00 或 1.00"
            f"（{degeneracy['extreme_share']:.1%}），弃权 {degeneracy['abstained']} 条"
        )
    best = metrics.get("best_point")
    if best is not None:
        lines.append(
            f"- 曲线上最宽松的一点：t={best['threshold']:.2f} "
            f"漏检 {best['false_pass_rate']:.1%} / 误杀 {best['false_fail_rate']:.1%}"
        )
    versus = metrics.get("judge_vs_rule")
    if versus is not None:
        lines.append(
            f"- 相对规则：一致 {versus['agreement']:.1%}"
            f"（判断者多判 {versus['judge_only_decided']} 条、规则多判 {versus['rule_only_decided']} 条）"
        )
    lines += [
        "",
        "| 阈值 | 漏检率 (false pass) | 误杀率 (false fail) | 弃权 | 漏检/攻击 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for point in report.get("points", []):
        fpr = point.get("false_pass_rate")
        ffr = point.get("false_fail_rate")
        cvr = point.get("cannot_verify_rate")
        lines.append(
            f"| {point['threshold']:.2f} | "
            f"{'—' if fpr is None else f'{fpr:.1%}'} | "
            f"{'—' if ffr is None else f'{ffr:.1%}'} | "
            f"{'—' if cvr is None else f'{cvr:.1%}'} | "
            f"{point['false_pass_count']}/{point['attacks']} |"
        )
    point = operating_point(report)
    lines += [
        "",
        (
            f"**推荐工作点：阈值 {point['threshold']:.2f}** —— "
            f"漏检 {point['false_pass_rate']:.1%}、误杀 {point['false_fail_rate']:.1%}。"
            if point
            else "**没有阈值能做到零漏检** —— 这条门单独用不够，需要规则兜底。"
        ),
    ]
    advice = report.get("recommendation") or {}
    if advice:
        label = {
            "rule_wins": "结论：**用规则，不要用模型门**",
            "judge_wins": "结论：**这回模型门赢**",
            "unusable": "结论：**这条门不可用**（概率是噪声）",
            "no_rule_baseline": "结论：**没有规则基线，只能答「准不准」**",
        }.get(advice.get("verdict"), f"结论：{advice.get('verdict')}")
        lines.append("")
        lines.append(label)
        for reason in advice.get("reasons", []):
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# A second criterion: "did this change stay inside its declared scope?"
#
# The citation criterion above was decided by a rule whenever the capture was
# in hand. This one asks the same question of a *judge* instead: given a scope
# the author declared and the files they actually touched, is every changed
# file inside the scope? The labels are built the same way as the URL corpus --
# real commits plus synthetic scope extensions -- so the curve is comparable.
#
# The point of running it is to test whether the "probability collapses to 0/1"
# result was a fluke of the URL task or a property of decidable criteria.
# ---------------------------------------------------------------------------

SCOPE_QUESTION = "is every changed file inside the declared scope?"

_SCOPE_SYSTEM = (
    "You audit a code change against the scope its author declared. Each request "
    "gives you the declared scope and the files that the change actually "
    "touched. Answer exactly one bounded question and nothing else: {question} "
    "Reply with a single integer from 0 to 100. 100 means you are certain every "
    "changed file is inside the declared scope; 0 means you are certain at least "
    "one is outside. No words, no units."
)

#: The lossy view: the file list is withheld and only the diff stat remains.
#: This is the scope analogue of the blind citation judge -- the gate sees that
#: *something* changed and roughly how much, but not *what*.
_SCOPE_LOSSY_SYSTEM = (
    "You audit a code change against the scope its author declared. Each request "
    "gives you the declared scope and a summary of the change (how many files, "
    "insertions and deletions). The list of changed files is NOT available. "
    "Answer exactly one bounded question and nothing else: {question} Reply with "
    "a single integer from 0 to 100. 100 means you are certain every changed "
    "file is inside the declared scope; 0 means you are certain at least one is "
    "outside. No words, no units."
)

#: Top-level areas that exist in this repo, used to synthesise a file that is
#: unambiguously outside any commit's declared scope.
FOREIGN_AREAS = (
    "harness",
    "tasks",
    "tests",
    "docs",
    "scripts",
    "experiments",
    "config",
    "tools",
)


def areas_of(files: Iterable[str]) -> list[str]:
    """Top-level directory of each path, deduplicated and sorted."""
    return sorted({path.split("/", 1)[0] for path in files if path})


def in_scope(declared: Iterable[str], files: Iterable[str]) -> bool:
    """The ground truth: every changed file is inside a declared area."""
    allowed = set(declared)
    return all((path.split("/", 1)[0] in allowed) for path in files)


def samples_from_commits(
    records: list[dict[str, Any]],
    *,
    max_per_kind: int = 8,
) -> list[dict[str, Any]]:
    """Turn commit records into labelled scope samples. Pure function, no git."""
    samples: list[dict[str, Any]] = []
    counts = {"legit": 0, "scope_extension": 0, "foreign_file": 0}

    def _add(kind: str, sha: str, subject: str, declared: list[str], files: list[str], note: str) -> None:
        if counts[kind] >= max_per_kind:
            return
        counts[kind] += 1
        samples.append(
            {
                "sample_id": f"scope-{kind}-{counts[kind]}-{sha[:8]}",
                "kind": kind,
                "expect": ve.EXPECT_PASS if in_scope(declared, files) else ve.EXPECT_NOT_PASS,
                "note": note,
                "sha": sha,
                "subject": subject,
                "declared": list(declared),
                "changed": list(files),
            }
        )

    for record in records:
        files = [f for f in (record.get("files") or []) if f]
        if not files:
            continue
        sha = str(record.get("sha") or "")
        subject = str(record.get("subject") or "")
        areas = areas_of(files)

        # The honest case: the author declared everything they touched.
        _add("legit", sha, subject, areas, files, "declared = touched")

        # A scope that covers only the first area: the rest is out of scope.
        if len(areas) >= 2:
            _add(
                "scope_extension",
                sha,
                subject,
                areas[:1],
                files,
                f"declared only {areas[0]}/ but the change reached {', '.join(areas[1:])}/",
            )

        # A file that no reasonable scope could contain.
        foreign = next((area for area in FOREIGN_AREAS if area not in areas), None)
        if foreign is not None:
            _add(
                "foreign_file",
                sha,
                subject,
                areas,
                files + [f"{foreign}/unrelated-injected-file.py"],
                f"a {foreign}/ file appears in a change scoped to {', '.join(areas)}/",
            )
    return samples


def _run_git(repo: str | Any, args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise HarnessError("git is not installed") from exc
    if proc.returncode != 0:
        raise HarnessError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def commits_in(repo: str | Any, *, limit: int = 120) -> list[dict[str, Any]]:
    """Recent non-merge commits with the files they touched."""
    out = _run_git(repo, ["log", f"-n{limit}", "--no-merges", "--numstat", "--pretty=format:@@%H\t%s"])
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in out.splitlines():
        if line.startswith("@@"):
            head = line[2:]
            sha, _, subject = head.partition("\t")
            current = {"sha": sha, "subject": subject, "files": []}
            records.append(current)
            continue
        if current is None or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 3 and parts[2]:
            current["files"].append(parts[2])
    return records


def build_scope_corpus(
    repo: str | Any,
    *,
    limit: int = 120,
    max_per_kind: int = 8,
) -> dict[str, Any]:
    records = commits_in(repo, limit=limit)
    return {
        "days": [f"repo:{Path(repo).name}"],
        "samples": samples_from_commits(records, max_per_kind=max_per_kind),
    }


def scope_probability_judge(
    provider: Any,
    *,
    max_tokens: int = 512,
    lossy: bool = False,
) -> Callable[[dict[str, Any]], float | None]:
    """Judge the scope question. ``lossy=True`` withholds the file list."""
    system_template = _SCOPE_LOSSY_SYSTEM if lossy else _SCOPE_SYSTEM
    usage: dict[str, Any] = {
        "calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "cost_known": False,
        "abstentions": 0,
    }

    def judge(sample: dict[str, Any]) -> float | None:
        declared = ", ".join(f"{area}/" for area in sample.get("declared") or [])
        files = sample.get("changed") or []
        if lossy:
            # Only totals. The per-area breakdown would leak the answer: if the
            # summary named the areas, membership would be decidable again and
            # the "lossy" arm would just be the decidable arm with extra words.
            user = (
                f"declared scope:\n{declared}\n\n"
                f"change summary (file list and per-area breakdown withheld): "
                f"{len(files)} files changed\n"
            )
        else:
            listing = "\n".join(files)
            user = f"declared scope:\n{declared}\n\nchanged files ({len(files)}):\n{listing}"
        usage["calls"] += 1
        try:
            response = provider.chat(
                [
                    {"role": "system", "content": system_template.format(question=SCOPE_QUESTION)},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:  # noqa: BLE001
            usage["abstentions"] += 1
            return ABSTAIN
        usage["tokens_in"] += int(response.tokens_in or 0)
        usage["tokens_out"] += int(response.tokens_out or 0)
        if response.cost_usd is not None:
            usage["cost_usd"] += float(response.cost_usd)
            usage["cost_known"] = True
        probability = _parse_probability(response.text or "")
        if probability is None:
            usage["abstentions"] += 1
        return probability

    judge.usage = usage  # type: ignore[attr-defined]
    judge.question = SCOPE_QUESTION  # type: ignore[attr-defined]
    return judge


def scope_probabilities(
    corpus: dict[str, Any],
    judge: Callable[[dict[str, Any]], float | None],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    total = len(corpus.get("samples", []))
    for index, sample in enumerate(corpus.get("samples", []), start=1):
        out[sample["sample_id"]] = judge(sample)
        if progress is not None:
            progress(index, total)
    return out


def scope_rule_probabilities(corpus: dict[str, Any]) -> dict[str, float | None]:
    """The deterministic baseline: 1.0 in scope, 0.0 out of scope.

    Scored through the same ``score_at`` path as the judge so the comparison is
    like for like rather than two different code paths.
    """
    return {
        sample["sample_id"]: (1.0 if in_scope(sample["declared"], sample["changed"]) else 0.0)
        for sample in corpus.get("samples", [])
    }


def citation_rule_probabilities(
    corpus: dict[str, Any], *, root: Any | None = None
) -> dict[str, float | None]:
    """The deterministic citation gate expressed on the same 0/1 scale.

    PASS maps to 1.0 and FAIL to 0.0 so it can be compared with the judge through
    one code path. CANNOT_VERIFY stays ``None`` -- the rule is not allowed to
    guess either.
    """
    known_cache: dict[str, set[str]] = {}
    out: dict[str, float | None] = {}
    for sample in corpus.get("samples", []):
        day = sample["day"]
        if day not in known_cache:
            known_cache[day] = ve._raw_urls(day, root=root)
        decision = ve.typed_matcher(sample["url"], known_cache[day])
        if decision == Decision.PASS.value:
            out[sample["sample_id"]] = 1.0
        elif decision == Decision.FAIL.value:
            out[sample["sample_id"]] = 0.0
        else:
            out[sample["sample_id"]] = None
    return out


def rule_probabilities(
    criterion: str, corpus: dict[str, Any], *, root: Any | None = None
) -> dict[str, float | None] | None:
    """The free baseline for a criterion, or ``None`` when it cannot be built.

    ``None`` is a real answer, not a failure: a report reused from before this
    field existed carries only what the judge saw, and inventing a rule baseline
    for it would be worse than saying "not comparable".
    """
    samples = corpus.get("samples") or []
    if criterion == "scope":
        if not samples or not all("declared" in s and "changed" in s for s in samples):
            return None
        return scope_rule_probabilities(corpus)
    if not samples or not all("url" in s and "day" in s for s in samples):
        return None
    return citation_rule_probabilities(corpus, root=root)


__all__ = [
    "ABSTAIN",
    "DEFAULT_QUESTION",
    "FOREIGN_AREAS",
    "SCOPE_QUESTION",
    "areas_of",
    "build_scope_corpus",
    "citation_rule_probabilities",
    "commits_in",
    "corpus_from_report",
    "best_point",
    "cost_profile",
    "degeneracy_stats",
    "in_scope",
    "recommendation",
    "rule_probabilities",
    "rule_agreement",
    "separation_auc",
    "markdown",
    "operating_point",
    "parse_thresholds",
    "probabilities",
    "probability_judge",
    "samples_from_commits",
    "score_at",
    "scope_probabilities",
    "scope_probability_judge",
    "scope_rule_probabilities",
    "sweep",
    "verdict_for",
]
