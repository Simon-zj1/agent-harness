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
) -> dict[str, Any]:
    points = [
        score_at(corpus, probs, threshold=float(t), band=band)
        for t in thresholds
    ]
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge": judge_name,
        "question": question,
        "blind": blind,
        "band": band,
        "days": corpus.get("days", []),
        "samples": len(corpus.get("samples", [])),
        "corpus_fingerprint": ve.corpus_fingerprint(corpus),
        "usage": usage,
        "points": points,
    }


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


__all__ = [
    "ABSTAIN",
    "DEFAULT_QUESTION",
    "FOREIGN_AREAS",
    "SCOPE_QUESTION",
    "areas_of",
    "build_scope_corpus",
    "commits_in",
    "in_scope",
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
