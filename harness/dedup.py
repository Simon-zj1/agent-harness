"""Cross-source duplicate detection: cheap first, model only in the grey band.

The pipeline used to have no duplicate check at all: the same story could be
written into `agent-engineering` and `ai-productivity` on the same day and
nothing noticed. This module adds the check in two tiers, which is the only
shape that stays affordable:

    clear duplicate   (score >= 0.86)  deterministic, no model, blocks
    grey band         (0.60 - 0.86)    model judges it, when a model is available
    distinct          (score <  0.60)  ignored

The grey band exists because editing duplicates is a judgement call that
string similarity cannot make ("OpenAI pauses training" vs "OpenAI's next model
delayed" may or may not be one story). Only those pairs cost a model call, and
the cost is capped per run.

Nothing here decides on its own to hide an item: every drop carries the pair
score and the reason, and unresolved pairs are recorded as unjudged rather than
silently kept or dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .decisions import Decision

DUP_SCORE = 0.86
BORDER_SCORE = 0.60

#: Wrappers and lead-ins that carry no signal for identity.
_PREFIXES = (
    "消息称",
    "报道称",
    "据报道",
    "据外媒",
    "官方：",
    "官方:",
    "突发",
    "独家",
    "解读：",
    "解读:",
)
_PUNCT = re.compile(r"[\s\u3000·—–\-—:：,，.。;；!！?？'\"“”‘’()（）\[\]【】《》<>/\\|*#`~+=]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9.+\-]*")


def normalize_title(text: str) -> str:
    """Lowercase, strip punctuation and the usual news lead-ins."""
    cleaned = _PUNCT.sub("", (text or "").strip().lower())
    for prefix in _PREFIXES:
        if cleaned.startswith(_PUNCT.sub("", prefix.lower())):
            cleaned = cleaned[len(_PUNCT.sub("", prefix.lower())) :]
    return cleaned


def tokens(text: str) -> set[str]:
    """CJK character bigrams plus latin words — cheap, no tokenizer dependency."""
    cleaned = _PUNCT.sub(" ", (text or "").lower())
    out: set[str] = {match.group(0) for match in _LATIN.finditer(cleaned) if len(match.group(0)) > 1}
    for run in re.findall(r"[\u4e00-\u9fff]+", cleaned):
        if len(run) == 1:
            out.add(run)
        for index in range(len(run) - 1):
            out.add(run[index : index + 2])
    return out


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


@dataclass
class Candidate:
    """One item, reduced to what duplicate detection needs."""

    location: str
    section: str
    group: str | None
    title_zh: str = ""
    title_en: str = ""
    body_zh: str = ""
    body_en: str = ""
    sources: list[int] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    rank: int = 0

    @property
    def label(self) -> str:
        return (self.title_zh or self.title_en or self.location)[:70]


@dataclass
class Pair:
    a: Candidate
    b: Candidate
    score: float
    reasons: list[str] = field(default_factory=list)
    verdict: str = "distinct"  # duplicate | borderline | distinct

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "verdict": self.verdict,
            "reasons": self.reasons,
            "a": {"location": self.a.location, "title": self.a.label},
            "b": {"location": self.b.location, "title": self.b.label},
        }


def _title_score(left: str, right: str) -> tuple[float, list[str]]:
    a, b = normalize_title(left), normalize_title(right)
    if not a or not b:
        return 0.0, []
    if a == b:
        return 0.95, ["normalised titles identical"]
    j = _jaccard(tokens(left), tokens(right))
    if j >= 0.8:
        return 0.85, [f"title token overlap {j:.2f}"]
    if j >= 0.65:
        return 0.72, [f"title token overlap {j:.2f}"]
    if j >= 0.5:
        return 0.62, [f"title token overlap {j:.2f}"]
    if j >= 0.35:
        return 0.5, [f"title token overlap {j:.2f}"]
    return 0.2 * j, []


def pair_score(a: Candidate, b: Candidate) -> tuple[float, list[str]]:
    """Score one pair in [0, 1] plus the reasons that produced the score."""
    score, reasons = _title_score(a.title_zh, b.title_zh)
    en_score, en_reasons = _title_score(a.title_en, b.title_en)
    if en_score > score:
        score, reasons = en_score, en_reasons

    shared = set(a.urls) & set(b.urls)
    if shared:
        reasons.append(f"shares cited URL ({len(shared)})")
        title_overlap = max(
            _title_overlap(a.title_zh, b.title_zh), _title_overlap(a.title_en, b.title_en)
        )
        # Two items supported by the same fetched document are suspicious, but
        # whether they are one story depends on where they sit:
        #
        #   same section + meaningful overlap  → provably one entry, drop it
        #   cross section + near-identical     → provably one entry, drop it
        #   cross section + partial overlap    → a judgement call: a roundup that
        #                                        mentions a repo is not the same
        #                                        thing as the repo's own card
        #
        # Critical: the shared source may promote a pair *into* the grey band,
        # never over the hard-duplicate line. Letting a bonus decide a drop was
        # the first version of this function, and on real data it auto-dropped
        # 09-28's roundup in favour of one of the repos it summarised.
        if a.section == b.section and title_overlap >= 0.4:
            score = max(score, 0.9)
            reasons.append(
                f"same section, shared source, title overlap {title_overlap:.2f}"
            )
        elif a.section != b.section and title_overlap >= 0.75:
            score = max(score, 0.9)
            reasons.append(
                f"cross section but near-identical title ({title_overlap:.2f}) "
                "with shared source"
            )
        elif score < BORDER_SCORE and title_overlap >= 0.3:
            score = 0.66
            reasons.append(
                f"shared source with title overlap {title_overlap:.2f} — needs a judgement"
            )

    if score < BORDER_SCORE:
        body_j = _jaccard(tokens(a.body_zh), tokens(b.body_zh))
        body_j_en = _jaccard(tokens(a.body_en), tokens(b.body_en))
        body_j = max(body_j, body_j_en)
        if body_j >= 0.6:
            score, reasons = 0.62, reasons + [f"body token overlap {body_j:.2f}"]

    return score, reasons


def _title_overlap(left: str, right: str) -> float:
    a, b = normalize_title(left), normalize_title(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return _jaccard(tokens(left), tokens(right))


def compare(candidates: Iterable[Candidate]) -> "DedupPlan":
    """All pairs at or above the grey band, worst-first, wrapped in a plan."""
    items = sorted(candidates, key=lambda c: c.rank)
    pairs: list[Pair] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            score, reasons = pair_score(left, right)
            if score < BORDER_SCORE:
                continue
            verdict = (
                "duplicate"
                if score >= DUP_SCORE
                else ("borderline" if score >= BORDER_SCORE else "distinct")
            )
            pairs.append(Pair(a=left, b=right, score=score, reasons=reasons, verdict=verdict))
    pairs.sort(key=lambda pair: pair.score, reverse=True)
    return DedupPlan(pairs=pairs)


@dataclass
class DedupPlan:
    pairs: list[Pair] = field(default_factory=list)

    @property
    def duplicates(self) -> list[Pair]:
        return [pair for pair in self.pairs if pair.verdict == "duplicate"]

    @property
    def borderline(self) -> list[Pair]:
        return [pair for pair in self.pairs if pair.verdict == "borderline"]


@dataclass
class DedupOutcome:
    dropped: list[dict[str, Any]] = field(default_factory=list)
    judged_distinct: list[dict[str, Any]] = field(default_factory=list)
    unjudged: list[dict[str, Any]] = field(default_factory=list)
    judge_calls: int = 0

    @property
    def dropped_locations(self) -> set[str]:
        return {entry["location"] for entry in self.dropped}

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped": self.dropped,
            "judged_distinct": self.judged_distinct,
            "unjudged": self.unjudged,
            "judge_calls": self.judge_calls,
        }


Judge = Callable[[Candidate, Candidate], Decision]


def resolve(plan: DedupPlan, *, judge: Judge | None = None, max_judge_calls: int = 8) -> DedupOutcome:
    """Turn pairs into drops.

    Clear duplicates always drop (the later item of the pair). Grey-band pairs
    drop only when a judge says they are the same story; without a judge they
    are recorded as unjudged so the uncertainty is visible instead of being
    rounded to either answer.
    """
    outcome = DedupOutcome()
    dropped: set[str] = set()

    for pair in plan.duplicates:
        keep, drop = (pair.a, pair.b) if pair.a.rank <= pair.b.rank else (pair.b, pair.a)
        if drop.location in dropped:
            continue
        dropped.add(drop.location)
        outcome.dropped.append(
            {
                "location": drop.location,
                "title": drop.label,
                "kept": keep.location,
                "score": round(pair.score, 3),
                "reasons": pair.reasons,
                "source": "deterministic",
            }
        )

    for pair in plan.borderline:
        keep, other = (pair.a, pair.b) if pair.a.rank <= pair.b.rank else (pair.b, pair.a)
        if other.location in dropped:
            continue
        if judge is None or outcome.judge_calls >= max_judge_calls:
            outcome.unjudged.append(pair.to_dict())
            continue
        outcome.judge_calls += 1
        decision = judge(keep, other)
        if decision is Decision.PASS:
            dropped.add(other.location)
            outcome.dropped.append(
                {
                    "location": other.location,
                    "title": other.label,
                    "kept": keep.location,
                    "score": round(pair.score, 3),
                    "reasons": pair.reasons,
                    "source": "model",
                }
            )
        elif decision is Decision.FAIL:
            outcome.judged_distinct.append(pair.to_dict())
        else:
            outcome.unjudged.append(pair.to_dict())

    return outcome


def build_candidates(view: Any, *, reference_urls: dict[int, str] | None = None) -> list[Candidate]:
    """Build dedup candidates from a `content_schema.ArticleView`."""
    urls_by_id = reference_urls or {}
    out: list[Candidate] = []
    for rank, item in enumerate(view.items):
        out.append(
            Candidate(
                location=item.location,
                section=item.section,
                group=item.group,
                title_zh=item.title_zh,
                title_en=item.title_en,
                body_zh=item.body_zh,
                body_en=item.body_en,
                sources=list(item.sources),
                urls=[urls_by_id[s] for s in item.sources if s in urls_by_id],
                rank=rank,
            )
        )
    return out


def reference_url_map(references: Iterable[dict[str, Any]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for ref in references:
        ref_id = ref.get("id")
        url = ref.get("url")
        if isinstance(ref_id, int) and isinstance(url, str) and url:
            out[ref_id] = url.rstrip("/")
    return out
