"""Measure the acceptance gate itself.

An acceptance gate that has never been measured is a belief. This module builds
a labelled corpus from real captures plus synthetic attacks, runs matchers over
it, and reports the two numbers that decide whether the gate is worth anything:

    false_pass_rate   attacks it let through (the costly direction)
    false_fail_rate   legitimate inputs it wrongly rejected (the noisy direction)

Matchers are compared on the same corpus so a claim like "typed decisions beat
the previous heuristic" is a measurement, not an opinion.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .decisions import Decision, canonical_url, classify_url

DEFAULT_DATA_DIR = "/Users/simon-zj/Documents/ChatGPT/daily-trends/data"

# Kinds, and what a correct gate must say about them.
KIND_LEGIT = "legit"
KIND_LEGIT_DRIFT = "legit_drift"
KIND_PREFIX_EXTENSION = "prefix_extension"
KIND_FABRICATED_SUFFIX = "fabricated_suffix"
KIND_UNRELATED = "unrelated"

EXPECT_PASS = "pass"
EXPECT_NOT_PASS = "not_pass"

_KIND_EXPECT = {
    KIND_LEGIT: EXPECT_PASS,
    KIND_LEGIT_DRIFT: EXPECT_PASS,
    KIND_PREFIX_EXTENSION: EXPECT_NOT_PASS,
    KIND_FABRICATED_SUFFIX: EXPECT_NOT_PASS,
    KIND_UNRELATED: EXPECT_NOT_PASS,
}


@dataclass
class Sample:
    sample_id: str
    kind: str
    url: str
    day: str
    expect: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def data_dir() -> Path:
    return Path(os.environ.get("DAILY_TRENDS_DATA_DIR", DEFAULT_DATA_DIR))


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_urls(day: str, *, root: Path | None = None) -> set[str]:
    raw = _load_json((root or data_dir()) / "raw" / f"{day}.json")
    from .validators import _collect_urls

    return {canonical_url(u) for u in _collect_urls(raw)}


def _cited_reference_urls(day: str, *, root: Path | None = None) -> list[str]:
    content = _load_json((root or data_dir()) / f"{day}.json")
    references = {ref.get("id"): ref for ref in content.get("references") or []}
    used: set[int] = set()
    for section in content.get("sections") or []:
        items = list(section.get("items") or [])
        for group in section.get("groups") or []:
            items.extend(group.get("items") or [])
        for item in items:
            used.update(s for s in (item.get("sources") or []) if isinstance(s, int))
    urls: list[str] = []
    for ref_id in sorted(used):
        ref = references.get(ref_id)
        if ref and ref.get("url"):
            urls.append(str(ref["url"]))
    return urls


def _drift_scheme(url: str) -> str:
    if url.startswith("https://"):
        return url.replace("https://", "http://", 1)
    if url.startswith("http://"):
        return url.replace("http://", "https://", 1)
    return url


def _drift_trailing_slash(url: str) -> str | None:
    """Add a trailing slash to the path.

    Only safe when there is no query string: inserting the slash before a query
    (`/x.html/?q` vs the real `/x.html?q`) produces a different URL shape rather
    than an enumerated drift, and labelling that "should pass" would be wrong.
    """
    if "?" in url:
        return None
    return url.rstrip("/") + "/"


def _drift_tracking(url: str) -> str:
    """Append a tracking param without clobbering an existing query string."""
    return url + ("&" if "?" in url else "?") + "utm_source=daily-trends"


def _drift_www(url: str) -> str:
    """Toggle the www prefix — in either direction, and only once."""
    if url.startswith(("https://www.", "http://www.")):
        return url.replace("://www.", "://", 1)
    return re.sub(r"^(https?://)", r"\1www.", url, count=1)


_DRIFT_BUILDERS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("scheme", _drift_scheme),
    ("trailing_slash", _drift_trailing_slash),
    ("tracking_params", _drift_tracking),
    ("www", _drift_www),
)


def build_corpus(
    days: Iterable[str],
    *,
    root: Path | None = None,
    max_per_kind_per_day: int = 25,
) -> dict[str, Any]:
    """Build a labelled corpus. Deterministic: same inputs, same corpus."""
    samples: list[Sample] = []
    used_days: list[str] = []

    for day in days:
        base = root or data_dir()
        if not (base / f"{day}.json").is_file() or not (base / "raw" / f"{day}.json").is_file():
            continue
        known = _raw_urls(day, root=root)
        used_days.append(day)

        legit = [u for u in _cited_reference_urls(day, root=root) if canonical_url(u) in known]
        for index, url in enumerate(legit[:max_per_kind_per_day]):
            samples.append(
                Sample(
                    sample_id=f"{day}-legit-{index:03d}",
                    kind=KIND_LEGIT,
                    url=url,
                    day=day,
                    expect=EXPECT_PASS,
                    note="a reference the published article actually cited",
                )
            )

        for index, url in enumerate(legit[:max_per_kind_per_day]):
            # Rotate the starting drift so kinds stay balanced, but fall through
            # when a drift is not applicable to this URL.
            chosen: tuple[str, str] | None = None
            for offset in range(len(_DRIFT_BUILDERS)):
                label, builder = _DRIFT_BUILDERS[(index + offset) % len(_DRIFT_BUILDERS)]
                candidate = builder(url)
                # The drift must be *textually* different (otherwise it is a
                # duplicate), but it is expected to canonicalise to the same
                # URL — that equivalence is what "enumerated drift" means.
                if candidate is not None and candidate != url:
                    chosen = (label, candidate)
                    break
            if chosen is None:
                continue
            label, drifted = chosen
            samples.append(
                Sample(
                    sample_id=f"{day}-drift-{index:03d}",
                    kind=KIND_LEGIT_DRIFT,
                    url=drifted,
                    day=day,
                    expect=EXPECT_PASS,
                    note=f"enumerated drift: {label}",
                )
            )

        # Attacks are built from real URLs, so the *prefix* is always genuine —
        # that is exactly why a prefix-based matcher waves them through.
        bases = sorted(u for u in known if len(u) > 20)[:max_per_kind_per_day]
        for index, real in enumerate(bases):
            attack = f"https://{real}/this-page-never-existed"
            if canonical_url(attack) in known:
                continue
            samples.append(
                Sample(
                    sample_id=f"{day}-ext-{index:03d}",
                    kind=KIND_PREFIX_EXTENSION,
                    url=attack,
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note="real URL + fabricated path",
                )
            )
        for index, real in enumerate(bases):
            attack = f"https://{real}-v99-fake"
            if canonical_url(attack) in known:
                continue
            samples.append(
                Sample(
                    sample_id=f"{day}-fab-{index:03d}",
                    kind=KIND_FABRICATED_SUFFIX,
                    url=attack,
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note="real URL + fabricated suffix",
                )
            )
        for index in range(min(5, max_per_kind_per_day)):
            samples.append(
                Sample(
                    sample_id=f"{day}-unrel-{index:03d}",
                    kind=KIND_UNRELATED,
                    url=f"https://example.invalid/{day}/invented-{index}",
                    day=day,
                    expect=EXPECT_NOT_PASS,
                    note="no relationship to the capture at all",
                )
            )

    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": used_days,
        "samples": [s.to_dict() for s in samples],
    }


def save_corpus(path: str | Path, corpus: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def load_corpus(path: str | Path) -> dict[str, Any]:
    return _load_json(Path(path))


def default_corpus_path() -> Path:
    from . import paths

    return paths.runs_dir() / "verification" / "corpus.json"


def subsample(corpus: dict[str, Any], per_kind: int) -> dict[str, Any]:
    """Deterministically trim a corpus, keeping the kinds balanced.

    Used to cap how many samples a paid matcher is asked to judge. Every matcher
    in a comparison must see the *same* subset, or the numbers are not comparable.
    """
    if per_kind <= 0:
        return corpus
    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for sample in corpus.get("samples", []):
        kind = sample["kind"]
        if seen.get(kind, 0) >= per_kind:
            continue
        seen[kind] = seen.get(kind, 0) + 1
        kept.append(sample)
    return {
        **corpus,
        "samples": kept,
        "subsampled": {"per_kind": per_kind, "from": len(corpus.get("samples", []))},
    }


def available_days(*, root: Path | None = None) -> list[str]:
    """Days that have both a composed article and its raw capture."""
    base = root or data_dir()
    raw_dir = base / "raw"
    if not raw_dir.is_dir():
        return []
    days: list[str] = []
    for raw_file in sorted(raw_dir.glob("*.json")):
        day = raw_file.stem
        if (base / f"{day}.json").is_file():
            days.append(day)
    return days


# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


def legacy_matcher(url: str, known: set[str]) -> str:
    """Reproduce the original boolean gate, faults included.

    Kept deliberately: an unflattering baseline measured on the same corpus is
    the only thing that turns "I improved it" into evidence.
    """
    from .validators import _url_in_raw

    return Decision.PASS.value if _url_in_raw(url, known) else Decision.FAIL.value


def typed_matcher(url: str, known: set[str]) -> str:
    return classify_url(url, known).decision.value


def llm_matcher(provider: Any, *, max_tokens: int = 8) -> Callable[[str, set[str]], str]:
    """Build a matcher backed by a chat provider.

    This is the LLM-as-judge baseline the industry is moving away from. It is
    wired the same way as the others so it lands in the same table — but it is
    never selected unless the caller passes --allow-llm.
    """

    usage: dict[str, Any] = {
        "calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "cost_known": False,
    }

    def match(url: str, known: set[str]) -> str:
        # The judge gets the *whole* capture. Truncating it would make the
        # baseline lose for the wrong reason and inflate the typed matcher's win.
        sample = "\n".join(sorted(known))
        try:
            response = provider.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You verify citations. Given a cited URL and the URLs that "
                            "were actually fetched, reply with exactly one word: "
                            "PASS if the cited URL was fetched, "
                            "FAIL if it clearly was not, "
                            "CANNOT_VERIFY if the list is insufficient to decide."
                        ),
                    },
                    {"role": "user", "content": f"cited: {url}\n\nfetched:\n{sample}"},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:  # noqa: BLE001 - one bad call must not abort the eval
            # An unavailable judge is an unverifiable judge, not a passing one.
            usage["calls"] += 1
            return Decision.CANNOT_VERIFY.value
        usage["calls"] += 1
        usage["tokens_in"] += int(response.tokens_in or 0)
        usage["tokens_out"] += int(response.tokens_out or 0)
        if response.cost_usd is not None:
            usage["cost_usd"] += float(response.cost_usd)
            usage["cost_known"] = True
        text = (response.text or "").strip().upper()
        for token, decision in (
            ("CANNOT_VERIFY", Decision.CANNOT_VERIFY),
            ("PASS", Decision.PASS),
            ("FAIL", Decision.FAIL),
        ):
            if token in text:
                return decision.value
        return Decision.CANNOT_VERIFY.value

    match.usage = usage  # type: ignore[attr-defined]
    return match


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    corpus: dict[str, Any],
    matcher: Callable[[str, set[str]], str],
    *,
    root: Path | None = None,
    matcher_name: str = "matcher",
) -> dict[str, Any]:
    """Run one matcher over the corpus and return a confusion matrix."""
    usage = getattr(matcher, "usage", None)
    if isinstance(usage, dict):
        for key in ("calls", "tokens_in", "tokens_out"):
            usage[key] = 0
        usage["cost_usd"] = 0.0
        usage["cost_known"] = False

    known_cache: dict[str, set[str]] = {}
    per_kind: dict[str, dict[str, int]] = {}
    verdicts: list[dict[str, Any]] = []
    attacks = 0
    attacks_passed = 0
    legit = 0
    legit_rejected = 0
    uncertain = 0

    for sample in corpus.get("samples", []):
        day = sample["day"]
        if day not in known_cache:
            known_cache[day] = _raw_urls(day, root=root)
        known = known_cache[day]
        verdict = matcher(sample["url"], known)
        expected = sample["expect"]

        bucket = per_kind.setdefault(sample["kind"], {})
        bucket[verdict] = bucket.get(verdict, 0) + 1
        if verdict == Decision.CANNOT_VERIFY.value:
            uncertain += 1

        if expected == EXPECT_NOT_PASS:
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
                "verdict": verdict,
                # Two different questions, deliberately kept apart:
                #   safe   — the gate neither leaked nor rejected real content
                #   strict — the gate committed to a definite verdict
                # CANNOT_VERIFY is safe against an attack but not strict; that is
                # exactly the trade-off the policy layer exists to price.
                "safe": (verdict == Decision.PASS.value)
                == (expected == EXPECT_PASS),
                "strict": (
                    verdict == Decision.PASS.value
                    if expected == EXPECT_PASS
                    else verdict == Decision.FAIL.value
                ),
            }
        )

    total = len(verdicts)
    return {
        "matcher": matcher_name,
        "samples": total,
        "attacks": attacks,
        "legit": legit,
        "usage": dict(usage) if isinstance(usage, dict) else None,
        "false_pass_rate": round(attacks_passed / attacks, 4) if attacks else None,
        "false_pass_count": attacks_passed,
        "false_fail_rate": round(legit_rejected / legit, 4) if legit else None,
        "false_fail_count": legit_rejected,
        "cannot_verify_rate": round(uncertain / total, 4) if total else None,
        "per_kind": per_kind,
        "verdicts": verdicts,
    }


def compare_matchers(
    corpus: dict[str, Any],
    matchers: dict[str, Callable[[str, set[str]], str]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    results = [
        score(corpus, matcher, root=root, matcher_name=name)
        for name, matcher in matchers.items()
    ]
    return {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": corpus.get("days", []),
        "samples": len(corpus.get("samples", [])),
        "matchers": results,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 验收闸门自身的测量",
        "",
        f"- 语料：{report['samples']} 条，来自 {', '.join(report['days']) or '（无）'}",
        f"- 生成时间：{report['generated_at']}",
        "",
        "| matcher | 漏检率 (false pass) | 误杀率 (false fail) | cannot_verify | 漏检/总数 | 成本 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in report["matchers"]:
        fpr = entry["false_pass_rate"]
        ffr = entry["false_fail_rate"]
        cvr = entry["cannot_verify_rate"]
        usage = entry.get("usage") or {}
        cost = "免费" if not usage.get("calls") else (
            f"${usage['cost_usd']:.4f}"
            if usage.get("cost_known")
            else f"{usage.get('tokens_in', 0)} tok in"
        )
        lines.append(
            f"| {entry['matcher']} | "
            f"{'—' if fpr is None else f'{fpr:.1%}'} | "
            f"{'—' if ffr is None else f'{ffr:.1%}'} | "
            f"{'—' if cvr is None else f'{cvr:.1%}'} | "
            f"{entry['false_pass_count']}/{entry['attacks']} | {cost} |"
        )
    lines += [
        "",
        "## 口径",
        "",
        "- 语料分五类：真实引用、可枚举漂移（应通过）；前缀延伸、后缀伪造、完全无关（应**不**通过）。",
        "- `漏检率` = 应拒绝的样本里被判为 PASS 的比例。这是有代价的方向：放行一条编造来源。",
        "- `误杀率` = 应通过的样本里被判为 FAIL 的比例。这是噪声方向：拦下真实内容。",
        "- `cannot_verify` 不计入漏检（它不是放行），但也**不是**通过——由任务策略决定是否阻断。",
        "- `成本`：确定性 matcher 免费；LLM 裁判按实际 token 计费，未配置单价时只报 token。",
        "- 判决的**果断程度**与准确率同等重要：同样是 0% 漏检，24 次 `FAIL` 和 16 次",
        "  `CANNOT_VERIFY` 对下游是完全不同的负担。",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "Sample",
    "build_corpus",
    "save_corpus",
    "load_corpus",
    "default_corpus_path",
    "available_days",
    "legacy_matcher",
    "typed_matcher",
    "llm_matcher",
    "score",
    "compare_matchers",
    "markdown",
]
