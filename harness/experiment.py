"""Evallab: run the same workload through several strategies and compare them."""

from __future__ import annotations

import csv
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .errors import HarnessError
from .ledger import Ledger, RunRow
from .runtime import RunOptions, Runner


@dataclass
class Arm:
    name: str
    description: str = ""
    compose_mode: str | None = None
    context_strategy: str = "full"
    memory_enabled: bool = False
    executor: str | None = None
    publish: bool = False
    requires: list[str] = field(default_factory=list)
    skip_steps: list[str] | None = None


@dataclass
class Experiment:
    name: str
    task: str
    description: str = ""
    arms: list[Arm] = field(default_factory=list)
    # Steps every arm skips unless it says otherwise. The default keeps
    # experiments off the shared stores (capture files, site repo) so that
    # A/B comparisons stay reproducible.
    skipped_steps: list[str] = field(default_factory=list)
    path: Path | None = None


def load(name: str | Path, *, root: Path | None = None) -> Experiment:
    base = Path(name)
    if not base.is_absolute() and not base.is_file():
        base = (root or paths.experiments_dir()) / f"{name}.toml"
    if not base.is_file():
        raise HarnessError(f"experiment not found: {base}")
    import tomllib

    raw = tomllib.loads(base.read_text(encoding="utf-8"))
    arms = [
        Arm(
            name=entry["name"],
            description=entry.get("description", ""),
            compose_mode=entry.get("compose_mode"),
            context_strategy=entry.get("context_strategy", "full"),
            memory_enabled=bool(entry.get("memory_enabled", False)),
            executor=entry.get("executor"),
            publish=bool(entry.get("publish", False)),
            requires=list(entry.get("requires", [])),
            skip_steps=entry.get("skip_steps"),
        )
        for entry in raw.get("arms", [])
    ]
    if not arms:
        raise HarnessError(f"experiment {base.stem} declares no arms")
    return Experiment(
        name=raw.get("name", base.stem),
        task=raw["task"],
        description=raw.get("description", ""),
        arms=arms,
        skipped_steps=list(raw.get("skip_steps", ["sync-site", "fetch", "publish"])),
        path=base,
    )


def run_experiment(
    experiment: Experiment,
    *,
    date: str,
    allow_llm: bool = False,
    allow_network: bool = False,
    dry_run: bool = True,
    only_arms: list[str] | None = None,
    runner: Runner | None = None,
) -> dict[str, Any]:
    runner = runner or Runner()
    experiment_id = f"{experiment.name}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    outcomes: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for arm in experiment.arms:
        if only_arms and arm.name not in only_arms:
            skipped.append({"arm": arm.name, "reason": "not selected by --arm"})
            continue
        blocked = _blocked_reason(
            arm, allow_llm=allow_llm, allow_network=allow_network, dry_run=dry_run
        )
        if blocked:
            skipped.append({"arm": arm.name, "reason": blocked})
            continue
        outcome = runner.run(
            RunOptions(
                task=experiment.task,
                date=date,
                dry_run=dry_run,
                force=True,
                executor=arm.executor,
                publish=arm.publish,
                trigger="experiment",
                experiment=experiment_id,
                arm=arm.name,
                compose_mode=arm.compose_mode,
                context_strategy=arm.context_strategy,
                memory_enabled=arm.memory_enabled,
                notify=False,
                skip_steps=(
                    arm.skip_steps
                    if arm.skip_steps is not None
                    else list(experiment.skipped_steps)
                ),
            )
        )
        outcomes.append(_arm_summary(arm, outcome.run))

    outdir = paths.runs_dir() / "experiments" / experiment_id
    outdir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": experiment.name,
        "experiment_id": experiment_id,
        "task": experiment.task,
        "date": date,
        "dry_run": dry_run,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "arms": outcomes,
        "skipped": skipped,
        "pareto": pareto(outcomes),
    }
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "report.md").write_text(_markdown(report), encoding="utf-8")
    _write_csv(outdir / "report.csv", outcomes)
    report["report_md"] = str(outdir / "report.md")
    report["report_csv"] = str(outdir / "report.csv")
    return report


def compare(
    experiment_id_prefix: str,
    *,
    ledger: Ledger | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Rebuild a report from the ledger, without re-running the arms.

    An experiment that already happened should not have to be paid for twice
    just to draw the Pareto frontier.
    """
    ledger = ledger or Ledger()
    runs = ledger.list_runs(experiment_prefix=experiment_id_prefix, limit=500)
    outcomes = []
    for row in runs:
        arm = Arm(name=row.arm or row.run_id, compose_mode=row.compose_mode)
        outcomes.append(_arm_summary(arm, row))

    report = {
        "experiment": experiment_id_prefix,
        "experiment_id": f"{experiment_id_prefix}-replay",
        "task": runs[0].task if runs else "",
        "date": runs[0].target_date if runs else "",
        "dry_run": bool(runs[0].dry_run) if runs else True,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "arms": outcomes,
        "skipped": [],
        "source": "ledger",
        "pareto": pareto(outcomes),
    }
    if write:
        outdir = paths.runs_dir() / "experiments" / f"{experiment_id_prefix}-compare"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (outdir / "report.md").write_text(_markdown(report), encoding="utf-8")
        _write_csv(outdir / "report.csv", outcomes)
        report["report_md"] = str(outdir / "report.md")
        report["report_csv"] = str(outdir / "report.csv")
    return report


def _blocked_reason(arm: Arm, *, allow_llm: bool, allow_network: bool, dry_run: bool) -> str:
    # Anything that spends tokens or spawns a paid agent must be opted into.
    if ("llm" in arm.requires or "executor" in arm.requires) and not allow_llm:
        return "needs --allow-llm (spends model tokens)"
    if "network" in arm.requires and not allow_network:
        return "needs --allow-network"
    if "executor" in arm.requires:
        if not arm.executor:
            return "declares an executor requirement but no executor"
        if dry_run:
            return "delegate arms need a non-dry-run (dry-run never spawns executors)"
    return ""


def _arm_summary(arm: Arm, run: RunRow) -> dict[str, Any]:
    validators = {v.get("name"): v for v in run.validators}
    verifiable = (validators.get("daily_trends_verifiable") or {}).get("metrics", {})
    structure = (validators.get("daily_trends_structure") or {}).get("metrics", {})
    required_ok = all(v.get("ok") for v in run.validators) if run.validators else None
    return {
        "arm": arm.name,
        "description": arm.description,
        "run_id": run.run_id,
        "status": run.status,
        "compose_mode": run.compose_mode,
        "context_strategy": run.context_strategy,
        "memory": run.memory_enabled,
        "executor": run.executor,
        "validators_ok": required_ok,
        "insights": structure.get("insights"),
        "repos": structure.get("repos"),
        "verifiable_ratio": verifiable.get("verifiable_ratio"),
        "tokens_in": run.tokens_in,
        "tokens_out": run.tokens_out,
        "cost_usd": run.cost_usd,
        "duration_ms": run.duration_ms,
        "tool_calls": run.tool_calls,
        "human_intervention": run.human_intervention,
        "skipped_steps": [
            step["id"] for step in run.steps if step.get("status") == "skipped"
        ],
        "failure": run.error,
    }


def _verification_score(arm: dict[str, Any]) -> float | None:
    """How much of the acceptance gate this arm actually satisfied, in 0..1."""
    ratio = arm.get("verifiable_ratio")
    if isinstance(ratio, (int, float)):
        return float(ratio)
    if arm.get("validators_ok") is True:
        return 1.0
    if arm.get("validators_ok") is False:
        return 0.0
    return None


def pareto(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost vs verification: which arms are not beaten on both axes at once.

    Without this, a cheap arm and a thorough arm look like an apples-to-oranges
    table row and the reader has to do the trade-off in their head. Cost falls
    back to tokens when no arm has a configured price.
    """
    use_usd = any(arm.get("cost_usd") is not None for arm in outcomes)
    # One arm can have several runs. A frontier over configurations needs one
    # point per configuration, so repeats are averaged rather than plotted as
    # separate (and therefore artificially dominant) points.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for arm in outcomes:
        score = _verification_score(arm)
        cost = (
            arm.get("cost_usd")
            if use_usd
            else (arm.get("tokens_in") or 0) + (arm.get("tokens_out") or 0)
        )
        if score is None or cost is None:
            continue
        grouped.setdefault(arm["arm"], []).append(
            {
                "cost": float(cost),
                "score": score,
                "status": arm.get("status"),
                "compose_mode": arm.get("compose_mode"),
            }
        )

    points: list[dict[str, Any]] = []
    for name, entries in grouped.items():
        points.append(
            {
                "arm": name,
                "runs": len(entries),
                "cost": sum(e["cost"] for e in entries) / len(entries),
                "verification_score": sum(e["score"] for e in entries) / len(entries),
                "status": entries[-1]["status"],
                "compose_mode": entries[-1]["compose_mode"],
            }
        )

    frontier: list[str] = []
    for point in points:
        dominated = any(
            other["cost"] <= point["cost"]
            and other["verification_score"] >= point["verification_score"]
            and (
                other["cost"] < point["cost"]
                or other["verification_score"] > point["verification_score"]
            )
            for other in points
            if other is not point
        )
        point["pareto_optimal"] = not dominated
        if not dominated:
            frontier.append(point["arm"])

    caveats: list[str] = []
    replay_on_frontier = [
        point["arm"]
        for point in points
        if point.get("pareto_optimal") and point.get("compose_mode") == "replay"
    ]
    if replay_on_frontier:
        caveats.append(
            "replay 臂复用已有内容，成本天然为 0，因此总是落在前沿上（"
            + "、".join(replay_on_frontier)
            + "）。它是基线，不是可选的生成方案；比较真实方案时请看其余点。"
        )
    if not use_usd:
        caveats.append(
            "没有臂配置单价，横轴退化为 token 总量；配置 provider 单价后才能比较真实成本。"
        )
    unscored = [
        arm["arm"] for arm in outcomes if _verification_score(arm) is None
    ]
    if unscored:
        shown = ", ".join(sorted(set(unscored))[:6])
        caveats.append(
            f"{len(unscored)} 个臂没有验证结果（在进入校验前就失败了），未画入图中：{shown}。"
            "排除它们会让前沿看起来比实际更干净。"
        )

    return {
        "axis": "cost_usd" if use_usd else "tokens",
        "frontier": sorted(frontier),
        "points": sorted(points, key=lambda item: (item["cost"], -item["verification_score"])),
        "caveats": caveats,
    }


def _row_summary(row: RunRow) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "experiment": row.experiment,
        "arm": row.arm,
        "status": row.status,
        "duration_ms": row.duration_ms,
        "tokens_in": row.tokens_in,
        "tokens_out": row.tokens_out,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 实验：{report['experiment']}",
        "",
        f"- 任务：`{report['task']}`",
        f"- 数据日期：{report['date']}",
        f"- 模式：{'dry-run（不发布）' if report['dry_run'] else '发布模式'}",
        f"- 生成时间：{report['generated_at']}",
        f"- 实验 ID：`{report['experiment_id']}`",
        "",
        "## 对比表",
        "",
        "| arm | 状态 | 校验 | insights | repos | 可核验率 | tokens(in/out) | 成本 | 耗时(s) | 工具调用 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in report["arms"]:
        tokens = f"{arm['tokens_in']}/{arm['tokens_out']}"
        cost = "—" if arm["cost_usd"] is None else f"${arm['cost_usd']:.4f}"
        duration = "—" if arm["duration_ms"] is None else f"{arm['duration_ms'] / 1000:.1f}"
        ratio = "—" if arm["verifiable_ratio"] is None else f"{arm['verifiable_ratio']:.2f}"
        lines.append(
            f"| {arm['arm']} | {arm['status']} | "
            f"{'ok' if arm['validators_ok'] else ('fail' if arm['validators_ok'] is False else '—')} | "
            f"{arm['insights'] if arm['insights'] is not None else '—'} | "
            f"{arm['repos'] if arm['repos'] is not None else '—'} | {ratio} | {tokens} | {cost} | "
            f"{duration} | {arm['tool_calls']} |"
        )
    if report["skipped"]:
        lines += ["", "## 跳过", ""]
        lines += [f"- {entry['arm']}：{entry['reason']}" for entry in report["skipped"]]
    frontier = report.get("pareto") or {}
    if frontier.get("points"):
        axis = "成本 (USD)" if frontier["axis"] == "cost_usd" else "token 总量"
        lines += [
            "",
            "## 成本—严格度 Pareto",
            "",
        f"横轴越小越好：{axis}；纵轴越大越好：可核验率（无该指标时取校验是否全过）。",
        "",
        "同一臂的多次运行取均值后作为图上一个点。",
        "",
        f"| arm | {axis} | 可核验率 | 帕累托最优 |",
            "| --- | --- | --- | --- |",
        ]
        for point in frontier["points"]:
            cost = (
                f"${point['cost']:.4f}"
                if frontier["axis"] == "cost_usd"
                else f"{int(point['cost'])}"
            )
            lines.append(
                f"| {point['arm']} | {cost} | {point['verification_score']:.2f} | "
                f"{'✅' if point.get('pareto_optimal') else ''} |"
            )
        lines += [
            "",
            f"- 前沿：{', '.join(frontier['frontier']) or '（无）'}",
            "- 前沿上的臂互不支配：想同时更便宜又更严格，只能换设计，不能在现有臂里挑。",
        ]
        for caveat in frontier.get("caveats", []):
            lines.append(f"- ⚠️ {caveat}")
    lines += [
        "",
        "## 口径",
        "",
        "- `可核验率`：正文引用的参考文献能在当日 raw 抓取结果里找到的比例（防编造闸门）。",
        "- `校验`：结构 + 引用 + 可核验性三组校验是否全部通过。",
        "- 臂默认跳过 `sync-site`/`fetch`/`publish`（跑同一份已抓取的 raw，保证各臂输入一致），"
        "因此状态常见为 `degraded`：产物有效，但确实少了现场抓取这一步。",
        "- 成本仅在配置了单价时才有值；未配置时只统计 token。",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, outcomes: list[dict[str, Any]]) -> None:
    if not outcomes:
        return
    keys = list(outcomes[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(outcomes)
