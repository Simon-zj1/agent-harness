"""Command line interface: run / status / report / memory / experiment / launchd / doctor."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from pathlib import Path

from . import (
    __version__,
    config as config_mod,
    gate_sweep,
    launchd,
    memory,
    paths,
    selection_eval,
    taskspec,
    validator_probes,
    verification_eval,
)
from .errors import AlreadyDone, ConfigError, HarnessError, LockBusy
from .experiment import compare as compare_experiment, load as load_experiment, run_experiment
from .ledger import Ledger
from .logutil import console
from .providers import get as get_provider
from .runtime import RESERVED_STEP_ENV, RunOptions, Runner, default_date

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_LOCKED = 3


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    handler = getattr(args, "_handler")
    try:
        return handler(args)
    except LockBusy as exc:
        console().error(f"blocked: {exc}")
        return EXIT_LOCKED
    except AlreadyDone as exc:
        console().info(str(exc))
        return EXIT_OK
    except (HarnessError, ConfigError) as exc:
        console().error(f"error: {exc}")
        return EXIT_FAIL
    except KeyboardInterrupt:
        console().error("interrupted")
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Local agent harness: own memory, own triggers, own acceptance criteria.",
    )
    parser.add_argument("--version", action="version", version=f"agent-harness {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a task once (idempotent per task+date)")
    run.add_argument("task")
    run.add_argument("--date", help="target date YYYY-MM-DD (defaults to the task's date_mode)")
    run.add_argument("--dry-run", action="store_true", help="never write publish targets")
    run.add_argument("--force", action="store_true", help="ignore the idempotency guard")
    run.add_argument("--executor", help="executor to hand heavy work to (codex/claude/builtin)")
    run.add_argument("--publish", dest="publish", action="store_true", default=None)
    run.add_argument("--no-publish", dest="publish", action="store_false")
    run.add_argument("--compose", dest="compose_mode", help="replay | llm | delegate")
    run.add_argument(
        "--context-strategy",
        default="full",
        choices=["full", "prefilter"],
        help="how raw capture is fed to the model",
    )
    run.add_argument("--memory", dest="memory_enabled", action="store_true", default=False)
    run.add_argument("--notify", dest="notify", action="store_true", default=None)
    run.add_argument("--no-notify", dest="notify", action="store_false")
    run.add_argument("--wait-lock", type=float, default=0.0, help="seconds to wait for the task lock")
    run.add_argument(
        "--skip-steps",
        default="",
        help="comma-separated step ids to skip (debugging; marks the run degraded)",
    )
    run.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment variable for task steps (repeatable)",
    )
    run.add_argument("--trigger", default="manual", help="manual | launchd | experiment")
    run.set_defaults(_handler=cmd_run)

    status = sub.add_parser("status", help="config, providers, memory and recent runs")
    status.add_argument("--task", default=None)
    status.set_defaults(_handler=cmd_status)

    runs = sub.add_parser("runs", help="list runs")
    runs.add_argument("--task")
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(_handler=cmd_runs)

    report = sub.add_parser("report", help="details of one run")
    report.add_argument("run_id")
    report.add_argument("--json", action="store_true")
    report.set_defaults(_handler=cmd_report)

    annotate = sub.add_parser("annotate", help="record human intervention or notes on a run")
    annotate.add_argument("run_id")
    annotate.add_argument("--intervened", action="store_true")
    annotate.add_argument("--notes")
    annotate.set_defaults(_handler=cmd_annotate)

    backup = sub.add_parser(
        "backup", help="snapshot the ledger and memory directory locally"
    )
    backup.add_argument("--out", help="destination directory (default: runs/backups/<timestamp>)")
    backup.set_defaults(_handler=cmd_backup)

    tasks = sub.add_parser("tasks", help="list declared tasks")
    tasks.set_defaults(_handler=cmd_tasks)

    mem = sub.add_parser("memory", help="file-first memory")
    mem_sub = mem.add_subparsers(dest="memory_command", required=True)
    mem_search = mem_sub.add_parser("search")
    mem_search.add_argument("query")
    mem_search.add_argument("--limit", type=int, default=10)
    mem_search.add_argument("--kind", choices=["runs", "notes"])
    mem_search.set_defaults(_handler=cmd_memory_search)
    mem_add = mem_sub.add_parser("add")
    mem_add.add_argument("--title", required=True)
    mem_add.add_argument("--body", required=True)
    mem_add.add_argument("--tags", default="")
    mem_add.add_argument("--source", default="")
    mem_add.set_defaults(_handler=cmd_memory_add)
    mem_promote = mem_sub.add_parser("promote")
    mem_promote.add_argument("run_id")
    mem_promote.add_argument("--title", required=True)
    mem_promote.add_argument("--body")
    mem_promote.add_argument("--tags", default="")
    mem_promote.set_defaults(_handler=cmd_memory_promote)
    mem_reindex = mem_sub.add_parser("reindex")
    mem_reindex.set_defaults(_handler=cmd_memory_reindex)

    exp = sub.add_parser("experiment", help="compare strategies on one workload")
    exp_sub = exp.add_subparsers(dest="experiment_command", required=True)
    exp_run = exp_sub.add_parser("run")
    exp_run.add_argument("name")
    exp_run.add_argument("--date", required=True)
    exp_run.add_argument("--allow-llm", action="store_true", help="allow arms that call a paid model")
    exp_run.add_argument("--allow-network", action="store_true")
    exp_run.add_argument(
        "--arm",
        action="append",
        default=None,
        help="only run these arms (repeatable); default is all non-blocked arms",
    )
    exp_run.add_argument(
        "--execute",
        action="store_true",
        help="run arms for real instead of dry-run (needed for delegate arms; arms "
        "still never publish unless they declare publish = true)",
    )
    exp_run.set_defaults(_handler=cmd_experiment_run)
    exp_list = exp_sub.add_parser("list")
    exp_list.set_defaults(_handler=cmd_experiment_list)
    exp_compare = exp_sub.add_parser(
        "compare", help="rebuild a report (with the cost/verification Pareto) from the ledger"
    )
    exp_compare.add_argument("prefix", help="experiment id prefix, e.g. daily-trends-compare")
    exp_compare.set_defaults(_handler=cmd_experiment_compare)

    lc = sub.add_parser("launchd", help="render/install the unattended trigger")
    lc_sub = lc.add_subparsers(dest="launchd_command", required=True)
    lc_render = lc_sub.add_parser("render")
    lc_render.add_argument("task")
    lc_render.add_argument("--out", help="write the plist here instead of ~/Library/LaunchAgents")
    lc_render.set_defaults(_handler=cmd_launchd_render)
    lc_install = lc_sub.add_parser("install")
    lc_install.add_argument("task")
    lc_install.add_argument("--load", action="store_true", help="also load it into launchd (starts scheduling)")
    lc_install.set_defaults(_handler=cmd_launchd_install)
    lc_status = lc_sub.add_parser("status")
    lc_status.add_argument("task")
    lc_status.set_defaults(_handler=cmd_launchd_status)

    doctor = sub.add_parser("doctor", help="check the environment this harness depends on")
    doctor.set_defaults(_handler=cmd_doctor)

    ver = sub.add_parser("verify", help="measure the acceptance gate itself")
    ver_sub = ver.add_subparsers(dest="verify_command", required=True)
    v_corpus = ver_sub.add_parser("corpus", help="build a labelled corpus from real captures")
    v_corpus.add_argument(
        "--days",
        action="append",
        default=None,
        help="YYYY-MM-DD (repeatable); default is every day with a capture",
    )
    v_corpus.add_argument("--out", help="where to write corpus.json")
    v_corpus.add_argument("--per-kind", type=int, default=25, help="cap per kind per day")
    v_corpus.set_defaults(_handler=cmd_verify_corpus)
    v_eval = ver_sub.add_parser("eval", help="score matchers against the corpus")
    v_eval.add_argument("--corpus", help="corpus.json (built on the fly when missing)")
    v_eval.add_argument(
        "--match",
        default="legacy,typed",
        help="comma-separated matchers: legacy, typed, llm",
    )
    v_eval.add_argument("--allow-llm", action="store_true", help="allow the paid llm-judge baseline")
    v_eval.add_argument("--llm-sample", type=int, default=40, help="cap samples sent to the judge")
    v_eval.add_argument(
        "--update-baseline",
        action="store_true",
        help="freeze this run as the regression baseline instead of comparing against it",
    )
    v_eval.set_defaults(_handler=cmd_verify_eval)
    v_sweep = ver_sub.add_parser(
        "sweep",
        help="Plan A: score a probability gate across thresholds (needs a model judge)",
    )
    v_sweep.add_argument("--corpus", help="corpus.json (built on the fly when missing)")
    v_sweep.add_argument(
        "--criterion",
        choices=("citation", "scope"),
        default="citation",
        help="citation: is this URL genuine evidence; scope: is this change inside its declared scope",
    )
    v_sweep.add_argument(
        "--repo",
        default=None,
        help="repo to read commits from for --criterion scope (default: this repo)",
    )
    v_sweep.add_argument(
        "--commits",
        type=int,
        default=120,
        help="how many recent commits --criterion scope reads",
    )
    v_sweep.add_argument(
        "--days",
        action="append",
        default=None,
        help="YYYY-MM-DD (repeatable); only used when no corpus is found",
    )
    v_sweep.add_argument("--allow-llm", action="store_true", help="required: this runs a paid judge")
    v_sweep.add_argument(
        "--per-kind",
        type=int,
        default=6,
        help="cap samples per kind per day sent to the judge (0 = no cap)",
    )
    v_sweep.add_argument(
        "--thresholds",
        default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
        help="comma list, or start:stop:step",
    )
    v_sweep.add_argument(
        "--band",
        type=float,
        default=0.0,
        help="width of the do-not-decide zone around the threshold (0 = none)",
    )
    v_sweep.add_argument(
        "--blind",
        action="store_true",
        help="citation: withhold the capture and judge plausibility from the URL alone; "
        "scope: withhold the file list and judge from the diff stat alone",
    )
    v_sweep.add_argument("--out", help="report directory")
    v_sweep.set_defaults(_handler=cmd_verify_sweep)
    v_probes = ver_sub.add_parser(
        "probes", help="adversarial probes for a validator's invariants"
    )
    v_probes.add_argument(
        "--group",
        default=None,
        help="probe group (default: all); currently: refund",
    )
    v_probes.set_defaults(_handler=cmd_verify_probes)
    v_content = ver_sub.add_parser(
        "content", help="content debt across days (read-only diagnosis)"
    )
    v_content.add_argument(
        "--days", action="append", default=None, help="YYYY-MM-DD (repeatable)"
    )
    v_content.set_defaults(_handler=cmd_verify_content)
    v_selection = ver_sub.add_parser(
        "selection",
        help="build a labelled sheet of what the pipeline chose, or score a filled one",
    )
    v_selection.add_argument(
        "--days",
        action="append",
        default=None,
        help="YYYY-MM-DD (repeatable); default is every day with a capture",
    )
    v_selection.add_argument("--sample", type=int, default=100, help="rows in the sheet")
    v_selection.add_argument("--out", help="directory for the sheet")
    v_selection.add_argument(
        "--score",
        help="score this already-labelled sheet (jsonl or csv) instead of building one",
    )
    v_selection.set_defaults(_handler=cmd_verify_selection)
    v_release = ver_sub.add_parser(
        "release",
        help="gate the site deploy: the newest day must pass every content gate",
    )
    v_release.add_argument(
        "--days", type=int, default=1, help="how many of the newest days to check"
    )
    v_release.set_defaults(_handler=cmd_verify_release)
    return parser


# -- commands --------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    runner = Runner()
    extra_env: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            console().error(f"--env expects KEY=VALUE, got {item!r}")
            return EXIT_USAGE
        key, value = item.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            console().error(f"--env has an invalid variable name: {key!r}")
            return EXIT_USAGE
        if key in RESERVED_STEP_ENV:
            console().error(f"--env cannot override reserved runtime variable {key!r}")
            return EXIT_USAGE
        extra_env[key] = value
    outcome = runner.run(
        RunOptions(
            task=args.task,
            date=args.date,
            dry_run=args.dry_run,
            force=args.force,
            executor=args.executor,
            publish=args.publish,
            trigger=args.trigger,
            compose_mode=args.compose_mode,
            context_strategy=args.context_strategy,
            memory_enabled=args.memory_enabled,
            notify=args.notify,
            wait_lock=args.wait_lock,
            skip_steps=[s.strip() for s in args.skip_steps.split(",") if s.strip()],
            extra_env=extra_env,
        )
    )
    log = console()
    if outcome.skipped:
        log.info(f"skip: {outcome.message}")
        return EXIT_OK
    run = outcome.run
    log.info(
        f"{run.task} {run.target_date}: {run.status} "
        f"(run {run.run_id}, {run.duration_ms} ms, published={run.published})"
    )
    for step in run.steps:
        log.info(f"  - {step['id']}: {step['status']} ({step.get('detail','')[:120]})")
    for result in run.validators:
        mark = "ok  " if result.get("ok") else "FAIL"
        log.info(f"  - validator {result['name']}: {mark} {result.get('detail','')}")
    if run.error:
        log.error(f"  failure: {run.failure_class}: {run.error}")
    return EXIT_OK if outcome.ok else EXIT_FAIL


def cmd_status(args: argparse.Namespace) -> int:
    paths.ensure_layout()
    log = console()
    config = config_mod.load()
    log.info(f"harness      : {paths.repo_root()} (version {__version__})")
    log.info(f"home         : {paths.home()}")
    log.info(f"config       : {config.path}")
    log.info(f"tasks        : {', '.join(taskspec.available()) or 'none'}")
    default_provider = config.provider()
    ok, note = get_provider(config).available()
    log.info(
        f"provider     : {default_provider.name} ({default_provider.model}) "
        f"{'ready' if ok else 'UNAVAILABLE'} — {note}"
    )
    stats = memory.stats()
    log.info(f"memory       : {stats['entries']} entries {stats['by_kind']} fts={stats['fts']}")

    ledger = Ledger()
    totals = ledger.totals(args.task)
    log.info(
        "runs         : {runs} total, {ok} ok, {failed} failed, {interventions} interventions".format(
            runs=totals.get("runs") or 0,
            ok=totals.get("ok") or 0,
            failed=totals.get("failed") or 0,
            interventions=totals.get("interventions") or 0,
        )
    )
    for row in ledger.list_runs(task=args.task, limit=5):
        log.info(
            f"  - {row.started_at} {row.task} {row.target_date} {row.status} "
            f"run={row.run_id} published={row.published}"
        )
    return EXIT_OK


def cmd_runs(args: argparse.Namespace) -> int:
    ledger = Ledger()
    for row in ledger.list_runs(task=args.task, limit=args.limit):
        console().info(
            f"{row.started_at}  {row.task:<14} {row.target_date}  {row.status:<9} "
            f"{'dry' if row.dry_run else '   '} {'pub' if row.published else '   '}  "
            f"{row.duration_ms or 0:>7}ms  {row.run_id}"
        )
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    ledger = Ledger()
    row = ledger.get_run(args.run_id)
    if row is None:
        console().error(f"unknown run: {args.run_id}")
        return EXIT_FAIL
    payload = {
        "run_id": row.run_id,
        "task": row.task,
        "target_date": row.target_date,
        "status": row.status,
        "executor": row.executor,
        "dry_run": row.dry_run,
        "published": row.published,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "duration_ms": row.duration_ms,
        "steps": row.steps,
        "validators": row.validators,
        "outputs": row.outputs,
        "metrics": row.metrics,
        "tokens": {"in": row.tokens_in, "out": row.tokens_out, "cost_usd": row.cost_usd},
        "tool_calls": row.tool_calls,
        "human_intervention": row.human_intervention,
        "failure_class": row.failure_class,
        "error": row.error,
        "notes": row.notes,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        log = console()
        log.info(f"{row.task} {row.target_date} → {row.status} ({row.run_id})")
        log.info(f"executor={row.executor} dry_run={row.dry_run} published={row.published}")
        log.info(f"duration={row.duration_ms} ms  tool_calls={row.tool_calls}")
        for step in row.steps:
            log.info(f"  - step {step['id']}: {step['status']} — {step.get('detail','')[:160]}")
        for result in row.validators:
            log.info(
                f"  - validator {result['name']}: {'ok' if result.get('ok') else 'FAIL'} "
                f"{result.get('metrics', {})}"
            )
        blocked = [r for r in row.validators if not r.get("ok")]
        if blocked:
            log.error("")
            log.error(f"被拦下：{len(blocked)} 个校验器未通过")
            for result in blocked:
                decision = result.get("decision") or ("fail" if not result.get("ok") else "—")
                failure_class = result.get("failure_class") or "—"
                log.error(f"  · {result['name']}  decision={decision} class={failure_class}")
                log.error(f"    {result.get('detail', '')}")
                for item in (result.get("evidence") or [])[:5]:
                    where = item.get("url") or item.get("ref") or ""
                    log.error(f"      - {item.get('ref', '')} {where}  {item.get('detail', '')}")
                extra = len(result.get("evidence") or []) - 5
                if extra > 0:
                    log.error(f"      … 另有 {extra} 条，见 validation-failures.json")
                if result.get("remediation"):
                    log.error(f"    修复建议：{result['remediation']}")
        if row.outputs:
            log.info(f"  outputs: {json.dumps(row.outputs, ensure_ascii=False)[:400]}")
        if row.error:
            log.error(f"  error: {row.error}")
    return EXIT_OK


def cmd_annotate(args: argparse.Namespace) -> int:
    ledger = Ledger()
    if ledger.get_run(args.run_id) is None:
        console().error(f"unknown run: {args.run_id}")
        return EXIT_FAIL
    ledger.annotate(
        args.run_id,
        intervened=True if args.intervened else None,
        notes=args.notes,
    )
    console().info(f"annotated {args.run_id}")
    return EXIT_OK


def cmd_backup(args: argparse.Namespace) -> int:
    paths.ensure_layout()
    out = (
        Path(args.out).expanduser()
        if args.out
        else paths.runs_dir() / "backups" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    if out.exists() and any(out.iterdir()):
        console().error(f"backup destination is not empty: {out}")
        return EXIT_FAIL
    out.mkdir(parents=True, exist_ok=True)

    ledger = Ledger()
    try:
        ledger.backup(out / "runs.db")
    finally:
        ledger.close()

    if paths.memory_dir().is_dir():
        shutil.copytree(paths.memory_dir(), out / "memory", dirs_exist_ok=True)
    if paths.runs_dir().is_dir():
        shutil.copytree(
            paths.runs_dir(),
            out / "runs",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "backups",
                "*.db",
                "*.db-wal",
                "*.db-shm",
                "*.db.backup",
            ),
        )
    console().info(f"backup written: {out}")
    return EXIT_OK


def cmd_tasks(args: argparse.Namespace) -> int:
    log = console()
    names = taskspec.available()
    if not names:
        log.info("no tasks declared")
    for name in names:
        task = taskspec.load(name)
        log.info(
            f"{task.name:<16} date_mode={task.date_mode:<9} steps={len(task.steps)} "
            f"validators={len(task.validators)} tools={len(task.allowed_tools)}"
        )
        if task.description:
            log.info(f"    {task.description}")
    return EXIT_OK


def cmd_memory_search(args: argparse.Namespace) -> int:
    results = memory.search(args.query, limit=args.limit, kind=args.kind)
    if not results:
        console().info("no matches")
        return EXIT_OK
    for entry in results:
        console().info(
            f"[{entry.get('kind')}] {entry.get('date')} {entry.get('title')}\n"
            f"    {paths.display(entry.get('path',''))}\n"
            f"    {str(entry.get('excerpt',''))[:200]}"
        )
    return EXIT_OK


def cmd_memory_add(args: argparse.Namespace) -> int:
    path = memory.add_note(
        args.title,
        args.body,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
        source=args.source,
    )
    console().info(f"note written: {path}")
    return EXIT_OK


def cmd_memory_promote(args: argparse.Namespace) -> int:
    path = memory.promote_run(
        args.run_id,
        ledger=Ledger(),
        title=args.title,
        body=args.body,
        tags=[t.strip() for t in args.tags.split(",") if t.strip()],
    )
    console().info(f"promoted to: {path}")
    return EXIT_OK


def cmd_memory_reindex(args: argparse.Namespace) -> int:
    count = memory.reindex()
    console().info(f"indexed {count} entries (fts={memory.fts_available()})")
    return EXIT_OK


def cmd_experiment_run(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.name)
    report = run_experiment(
        experiment,
        date=args.date,
            allow_llm=args.allow_llm,
            allow_network=args.allow_network,
            dry_run=not args.execute,
            only_arms=args.arm,
        )
    log = console()
    log.info(f"experiment {report['experiment_id']} finished")
    for arm in report["arms"]:
        log.info(
            f"  - {arm['arm']:<28} {arm['status']:<9} "
            f"validators={'ok' if arm['validators_ok'] else ('fail' if arm['validators_ok'] is False else '—')} "
            f"tokens={arm['tokens_in']}/{arm['tokens_out']} "
            f"duration={arm['duration_ms']}ms"
        )
    for entry in report["skipped"]:
        log.info(f"  - {entry['arm']:<28} skipped: {entry['reason']}")
    log.info(f"report: {report['report_md']}")
    return EXIT_OK


def cmd_experiment_list(args: argparse.Namespace) -> int:
    root = paths.experiments_dir()
    if not root.is_dir():
        console().info("no experiments declared")
        return EXIT_OK
    for path in sorted(root.glob("*.toml")):
        experiment = load_experiment(path)
        console().info(f"{experiment.name:<28} task={experiment.task} arms={len(experiment.arms)}")
    return EXIT_OK


def cmd_experiment_compare(args: argparse.Namespace) -> int:
    log = console()
    report = compare_experiment(args.prefix)
    if not report["arms"]:
        log.error(f"no runs recorded under experiment prefix {args.prefix!r}")
        return EXIT_FAIL
    log.info(f"rebuilt {len(report['arms'])} arm(s) from the ledger")
    frontier = report.get("pareto") or {}
    if frontier.get("points"):
        axis = "cost_usd" if frontier["axis"] == "cost_usd" else "tokens"
        for point in frontier["points"]:
            mark = "★" if point.get("pareto_optimal") else " "
            log.info(
                f"  {mark} {point['arm']:<28} {axis}={point['cost']:>10.2f} "
                f"verify={point['verification_score']:.2f}"
            )
        log.info(f"frontier: {', '.join(frontier['frontier']) or '（无）'}")
    log.info(f"report: {report['report_md']}")
    return EXIT_OK


def cmd_launchd_render(args: argparse.Namespace) -> int:
    task = taskspec.load(args.task)
    if args.out:
        xml = launchd.to_xml(task)
        Path(args.out).expanduser().write_text(xml, encoding="utf-8")
        console().info(f"plist written: {args.out}")
        return EXIT_OK
    sys.stdout.write(launchd.to_xml(task))
    return EXIT_OK


def cmd_launchd_install(args: argparse.Namespace) -> int:
    task = taskspec.load(args.task)
    path = launchd.install(task, load=args.load)
    console().info(f"plist installed: {path}")
    if not args.load:
        console().info(
            f"not loaded into launchd yet. enable it with: launchctl load {path}"
        )
    return EXIT_OK


def cmd_launchd_status(args: argparse.Namespace) -> int:
    task = taskspec.load(args.task)
    info = launchd.status(task)
    console().info(json.dumps(info, ensure_ascii=False))
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    log = console()
    problems: list[str] = []
    warnings: list[str] = []

    log.info(f"python      : {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        problems.append("python >= 3.11 required (tomllib)")
    for binary in ("git", "curl", "osascript", "plutil"):
        path = shutil.which(binary)
        log.info(f"{binary:<12}: {path or 'MISSING'}")
        if not path:
            problems.append(f"{binary} missing")
    for binary in ("codex", "claude"):
        path = shutil.which(binary)
        log.info(f"executor {binary:<3}: {path or 'not installed (optional)'}")

    config = config_mod.load()
    for name, provider in config.providers.items():
        ok, note = get_provider(config, name).available()
        log.info(f"provider {name:<8}: {'ready' if ok else 'UNAVAILABLE'} — {note}")
        if not ok and name == config.default_provider:
            problems.append(f"default provider {name} unavailable: {note}")

    log.info(f"memory fts  : {'yes' if memory.fts_available() else 'no (LIKE fallback)'}")
    tasks = taskspec.available()
    log.info(f"tasks       : {', '.join(tasks) or 'none'}")
    for name in tasks:
        task = taskspec.load(name)
        for step in task.steps:
            script = step.command[-1] if step.command else ""
            if script.endswith(".py"):
                candidate = task.dir / script
                if not candidate.is_file():
                    problems.append(f"{name}: step {step.id} script missing: {candidate}")
        for spec in task.validators:
            from .validators import names as validator_names

            if spec.name not in validator_names():
                problems.append(f"{name}: unknown validator {spec.name}")
        for entry in task.readable_paths:
            if "{run_dir}" in entry:
                continue
            try:
                resolved = task.path_for(entry, date=default_date(task))
            except Exception as exc:  # noqa: BLE001 - report, don't crash doctor
                problems.append(f"{name}: cannot resolve readable path {entry!r}: {exc}")
                continue
            if not resolved.exists():
                # 外部依赖缺失只算告警：另外两个任务不依赖它，而这个任务真正开跑时
                # 会在同一句话上失败。把它算成 doctor 失败，等于让人第一次 clone
                # 下来就看到「环境坏了」，而实际上只有这一个任务不可用。
                warnings.append(
                    f"{name}: 外部依赖缺失，该任务暂不可用: {resolved} "
                    f"(设任务的环境变量覆盖，或把依赖 clone 到该路径)"
                )
        if str((task.trigger or {}).get("type", "")).lower() == "launchd":
            try:
                trigger_status = launchd.status(task)
            except FileNotFoundError:
                log.info(f"launchd {name:<12}: unavailable on this platform")
            except Exception as exc:  # noqa: BLE001 - report, don't crash doctor
                problems.append(f"{name}: launchd status failed: {exc}")
            else:
                log.info(
                    f"launchd {name:<12}: "
                    f"{'loaded' if trigger_status.get('loaded') else 'not loaded'}"
                )

    ledger = Ledger()
    log.info(f"ledger      : {ledger.path}")
    integrity_findings = ledger.integrity_check()
    if integrity_findings:
        problems.append("ledger integrity: " + "; ".join(integrity_findings))
    else:
        log.info("ledger check: ok")
    spent = ledger.monthly_cost()
    limit = config.budget.monthly_limit_usd
    if limit is not None:
        log.info(
            f"monthly api : ${spent:.4f} / ${limit:.2f} "
            f"({'over budget' if spent >= float(limit) else 'ok'})"
        )
    else:
        log.info(f"monthly api : ${spent:.4f} (no limit configured)")
    try:
        backup_path = ledger.backup()
        log.info(f"ledger bkup : {backup_path}")
    except Exception as exc:  # noqa: BLE001 - doctor should report, not crash
        problems.append(f"ledger backup failed: {exc}")
    log.info(f"now         : {dt.datetime.now().astimezone().isoformat(timespec='seconds')}")

    if warnings:
        log.info("warnings:")
        for warning in warnings:
            log.info(f"  - {warning}")
    if problems:
        log.error("problems:")
        for problem in problems:
            log.error(f"  - {problem}")
        return EXIT_FAIL
    log.info("all checks passed" if not warnings else "all checks passed (有告警，见上)")
    return EXIT_OK


def cmd_verify_probes(args: argparse.Namespace) -> int:
    log = console()
    try:
        probes = validator_probes.all_probes(args.group)
    except KeyError as exc:
        log.error(str(exc))
        return EXIT_FAIL
    report = validator_probes.run_probes(probes)
    outdir = (
        paths.runs_dir() / "verification" / f"probes-{dt.datetime.now():%Y%m%d-%H%M%S}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(validator_probes.markdown(report), encoding="utf-8")
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for outcome in report["outcomes"]:
        log.info(
            f"  {'OK ' if outcome['correct'] else 'BAD'} {outcome['probe_id']:<26} "
            f"expect={outcome['expect']:<9} "
            f"{'blocked' if outcome['blocked'] else 'allowed':<8} {outcome['note']}"
        )
    log.info(
        f"probes={report['probes']} 漏判={report['missed']} 误拦={report['overblocked']}"
    )
    log.info(f"report: {outdir / 'report.md'}")
    return EXIT_OK if report["ok"] else EXIT_FAIL


def _required_validators(task: str = "daily-trends") -> set[str] | None:
    """Which gates block a release — read from the task declaration.

    `verify content` / `verify release` used to decide severity from their own
    hardcoded list while a run decided it from `task.toml.required`. Flipping a
    gate between warn and block therefore changed one path and not the other.
    """
    try:
        spec = taskspec.load(task)
    except Exception:  # noqa: BLE001 - a missing task must not break diagnosis
        return None
    return {validator.name for validator in spec.validators if validator.required}


def _task_tools_dir(task: str = "daily-trends") -> Path | None:
    """The tools checkout declared by the task — one source for every gate path."""
    try:
        spec = taskspec.load(task)
    except Exception:  # noqa: BLE001
        return None
    raw = spec.paths_table.get("tools_dir")
    if not raw:
        return None
    rendered = taskspec.render(raw, date=spec.date_mode)
    path = Path(rendered).expanduser()
    # NB: 必须用展开后的字符串拼路径。第一版用了未展开的 `raw`，于是返回值里带着
    # `${DAILY_TRENDS_DIR:-...}` 字面量，brief/depth/coverage 全都找不到模块。
    return (path if path.is_absolute() else (spec.dir / rendered)).resolve()


def cmd_verify_content(args: argparse.Namespace) -> int:
    log = console()
    days = args.days or verification_eval.available_days()
    if not days:
        log.error("no day has both a capture and a composed article")
        return EXIT_FAIL
    report = verification_eval.content_debt(
        days, required=_required_validators(), tools_dir=_task_tools_dir()
    )
    outdir = paths.runs_dir() / "verification" / "content-debt"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(
        verification_eval.content_debt_markdown(report), encoding="utf-8"
    )
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in report["rows"]:
        flag = "ok  " if row["clean"] else "FAIL"
        if row["clean"] and row.get("warnings"):
            flag = "ok* "
        log.info(
            f"  {flag} {row['day']}  ratio={row['verifiable_ratio']} "
            f"fail={row['fail']} orphans={row['orphan_references']} "
            f"brief={row.get('brief_entries')} depth={row.get('depth_limit_ratio')} "
            f"blockers={','.join(row['blockers']) or '-'}"
        )
    log.info(f"干净 {report['clean']}/{report['days']} 天")
    if report.get("clean_with_warnings"):
        log.info(
            f"ok* = 通过但有告警（{', '.join(report['warned'])}）：不拦发布，只记录"
        )
    log.info(f"report: {outdir / 'report.md'}")
    return EXIT_OK


def cmd_verify_release(args: argparse.Namespace) -> int:
    """The gate the site deploy calls before it publishes anything.

    Without this, every other check in this repository is a post-mortem: the
    harness can prove a day was bad, but only after it is already on the site.
    `AGENT_RELEASE_GATE=off` is the deliberate, visible way to publish anyway.
    """
    log = console()
    if os.environ.get("AGENT_RELEASE_GATE", "on").lower() == "off":
        log.info("release gate skipped: AGENT_RELEASE_GATE=off")
        return EXIT_OK

    # 按**成品**的日期审：即将发布的那一天。用 available_days()（要求 raw 也在）
    # 会让「最新一天缺 raw」时闸门自动退到前一天并放行（审计发现 1.1）。
    days = verification_eval.content_days()
    if not days:
        log.error("release gate: 找不到任何成品内容（data/<date>.json）")
        return EXIT_FAIL
    target = days[-max(1, args.days) :]
    report = verification_eval.content_debt(
        target, required=_required_validators(), tools_dir=_task_tools_dir()
    )
    # 无论通过与否都留档：以前只有失败才写 report.json，于是"通过但有告警"的
    # 那一次在磁盘上没有任何痕迹（审计发现 2.2）。
    outdir = paths.runs_dir() / "verification" / "release"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "report.md").write_text(
        verification_eval.content_debt_markdown(report), encoding="utf-8"
    )
    dirty = [row for row in report["rows"] if not row["clean"]]

    for row in report["rows"]:
        mark = "ok  " if row["clean"] else "FAIL"
        extra = " raw缺失" if row.get("raw_missing") else ""
        log.info(
            f"  {mark} {row['day']}  blockers={','.join(row['blockers']) or '-'}  "
            f"ratio={row['verifiable_ratio']}{extra}"
        )

    # 尺度：破坏页面/结构的问题拦发布；引用「来自别的某天」是产线用了陈旧素材，
    # 页面仍然可用，记为告警；只有「任何一天都找不到」的引用才是编造风险，拦发布。
    blocking: list[dict] = []
    warned: list[str] = []
    for row in dirty:
        hard = [name for name in row["blockers"] if name != "verifiable"]
        if "verifiable" in row["blockers"]:
            if row.get("raw_missing"):
                # 没有当日抓取 → 一条都核验不了，不是"用了昨天的素材"，拦发布。
                hard.append("verifiable")
                warned.append(f"{row['day']}: 缺少当日 raw 抓取，无法核验任何引用")
            elif (row.get("unknown_citations") or 0) > 0:
                hard.append("verifiable")
            else:
                warned.append(
                    f"{row['day']}: {row.get('stale_citations') or 0} 条引用来自其它日期的抓取"
                )
        if hard:
            blocking.append({**row, "hard_blockers": hard})

    for line in warned:
        log.info(f"  warn {line}")
    for row in report["rows"]:
        examples = row.get("coverage_examples") or []
        if examples or row.get("coverage_pool"):
            log.info(
                f"  cover {row['day']}: 高信号漏 {row.get('coverage_missed')} 条"
                f"（命中方向 {row.get('coverage_on_topic_missed')}、来源前 3 名 "
                f"{row.get('coverage_top3_missed')}）"
                + (f"；例：{'；'.join(examples[:3])}" if examples else "")
            )
    if not blocking:
        log.info(f"release gate: {', '.join(target)} 通过全部内容闸门")
        log.info(f"  report: {outdir / 'report.md'}")
        return EXIT_OK

    log.error(
        "release gate: 内容未通过验收，阻止发布 —— "
        + "; ".join(f"{row['day']}: {','.join(row['hard_blockers'])}" for row in blocking)
    )
    log.error(f"  详情见 {outdir / 'report.md'}")
    log.error("  修好内容后重试；确实要发布未过闸门的版本：AGENT_RELEASE_GATE=off")
    return EXIT_FAIL


def cmd_verify_selection(args: argparse.Namespace) -> int:
    """Build a labelled sheet, or score one a human has filled in."""
    log = console()

    if args.score:
        sheet = Path(args.score)
        rows = selection_eval.load_sheet(sheet)
        report = selection_eval.score_sheet(rows)
        outdir = sheet.parent
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        (outdir / f"score-{stamp}.md").write_text(
            selection_eval.markdown(report), encoding="utf-8"
        )
        (outdir / f"score-{stamp}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if report["labelled"] == 0:
            log.error(
                "0 条人工标注：评分需要先把 sheet 的 gold 列填成 select/reject/either。"
                "没有标注就没有证据，这里不会给数字。"
            )
            log.info(f"待标注：{sheet}（{report['sheet_rows']} 行）")
            return EXIT_FAIL
        pct = lambda value: "—" if value is None else f"{value * 100:.1f}%"  # noqa: E731
        log.info(
            f"标注 {report['labelled']} 条（两可 {report['either']}，未标注 {report['unlabelled']}）"
        )
        log.info(
            f"查准率 {pct(report['precision'])}  查全率 {pct(report['recall'])}  "
            f"准确率 {pct(report['accuracy'])}"
        )
        log.info(
            f"  TP={report['counts']['tp']} FP={report['counts']['fp']} "
            f"FN={report['counts']['fn']} TN={report['counts']['tn']}"
        )
        if report["labelled"] < 30:
            log.info("提示：标注少于 30 条时这些比率只是方向，不要当结论。")
        log.info(f"report: {outdir / f'score-{stamp}.md'}")
        return EXIT_OK

    days = args.days or selection_eval.available_days()
    if not days:
        log.error("no day has both a capture and a composed article")
        return EXIT_FAIL
    rows = selection_eval.build_sheet(days, sample=args.sample)
    if not rows:
        log.error("no candidates found; nothing to label")
        return EXIT_FAIL
    written = selection_eval.write_sheet(rows, Path(args.out) if args.out else None)
    buckets: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for row in rows:
        buckets[row.bucket] = buckets.get(row.bucket, 0) + 1
        kinds[row.kind] = kinds.get(row.kind, 0) + 1
    log.info(f"sheet: {len(rows)} 行，覆盖 {len(days)} 天")
    log.info(f"  bucket: {buckets}")
    log.info(f"  kind:   {kinds}")
    log.info(f"  {written['csv']}   ← 用 Excel 填 gold 列（select / reject / either）")
    log.info(f"  {written['jsonl']} ← 或者直接改 jsonl")
    log.info(
        "填完后评分：./agent verify selection --score "
        f"{written['csv']}"
    )
    log.info("draft 列是机器猜测，只作参考；评分只认 gold 列。")
    return EXIT_OK


def cmd_verify_corpus(args: argparse.Namespace) -> int:
    log = console()
    days = args.days or verification_eval.available_days()
    if not days:
        log.error("no day has both a capture and a composed article")
        return EXIT_FAIL
    corpus = verification_eval.build_corpus(days, max_per_kind_per_day=args.per_kind)
    target = Path(args.out) if args.out else verification_eval.default_corpus_path()
    verification_eval.save_corpus(target, corpus)
    from collections import Counter

    kinds = Counter(sample["kind"] for sample in corpus["samples"])
    log.info(f"corpus: {len(corpus['samples'])} samples over {len(days)} day(s)")
    for kind, count in sorted(kinds.items()):
        log.info(f"  - {kind:<20} {count}")
    log.info(f"written: {target}")
    return EXIT_OK


def cmd_verify_sweep(args: argparse.Namespace) -> int:
    """Plan A: score a per-evidence probability gate across thresholds.

    Separate from `verify eval` on purpose. That command compares matchers that
    already decided; this one decides *where the boundary should be*, which is
    the extra step a probability gate buys. The judge proposes a number and the
    threshold is swept in application code, so the model never sees a label.
    """
    log = console()
    if not args.allow_llm:
        log.error("verify sweep runs a paid model judge; pass --allow-llm to enable it")
        return EXIT_FAIL

    if args.criterion == "scope":
        repo = args.repo or os.environ.get("AGENT_REPO_ROOT") or Path.cwd()
        corpus = gate_sweep.build_scope_corpus(repo, limit=args.commits, max_per_kind=args.per_kind)
        if not corpus["samples"]:
            log.error(f"no commits to build a scope corpus from in {repo}")
            return EXIT_FAIL
        log.info(f"scope corpus: {len(corpus['samples'])} samples from {repo}")
        rule_probs = gate_sweep.scope_rule_probabilities(corpus)
    else:
        corpus_path = Path(args.corpus) if args.corpus else verification_eval.default_corpus_path()
        if corpus_path.is_file():
            corpus = verification_eval.load_corpus(corpus_path)
            log.info(f"corpus: {corpus_path} ({len(corpus.get('samples', []))} samples)")
        else:
            days = args.days or verification_eval.available_days()
            if not days:
                log.error("no corpus and no captures to build one from")
                return EXIT_FAIL
            corpus = verification_eval.build_corpus(days)
            verification_eval.save_corpus(corpus_path, corpus)
            log.info(f"corpus: built {len(corpus['samples'])} samples -> {corpus_path}")
        rule_probs = None
        before = len(corpus.get("samples", []))
        if args.per_kind:
            corpus = verification_eval.subsample(corpus, args.per_kind)
        if len(corpus.get("samples", [])) != before:
            log.info(f"judge budget: {len(corpus['samples'])}/{before} samples")

    agent_config = config_mod.load()
    provider = get_provider(agent_config)
    ok, detail = provider.available()
    if not ok:
        log.error(f"judge unavailable: {detail}")
        return EXIT_FAIL

    def _progress(done: int, total: int) -> None:
        if done == total or done % 10 == 0:
            log.info(f"  judging {done}/{total}")

    if args.criterion == "scope":
        judge = gate_sweep.scope_probability_judge(provider, lossy=args.blind)
        probs = gate_sweep.scope_probabilities(corpus, judge, progress=_progress)
    else:
        judge = gate_sweep.probability_judge(provider, blind=args.blind)
        probs = gate_sweep.probabilities(corpus, judge, progress=_progress)
    thresholds = gate_sweep.parse_thresholds(args.thresholds)
    if not thresholds:
        log.error(f"no thresholds parsed from {args.thresholds!r}")
        return EXIT_FAIL
    report = gate_sweep.sweep(
        corpus,
        probs,
        thresholds=thresholds,
        band=args.band,
        judge_name=provider.name,
        question=getattr(judge, "question", gate_sweep.DEFAULT_QUESTION),
        blind=args.blind,
        usage=getattr(judge, "usage", None),
    )
    report["criterion"] = args.criterion
    report["blind_or_lossy"] = bool(args.blind)
    if rule_probs is not None:
        # The deterministic baseline, scored through the same path so the
        # comparison is the same code and not a second implementation.
        report["rule"] = gate_sweep.score_at(corpus, rule_probs, threshold=0.5)
    outdir = (
        Path(args.out)
        if args.out
        else paths.runs_dir()
        / "verification"
        / f"gate-sweep-{args.criterion}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "report.md").write_text(gate_sweep.markdown(report), encoding="utf-8")
    # The judge calls are the expensive part; keep them so the curve can be
    # re-drawn at other thresholds for free.
    (outdir / "probabilities.json").write_text(
        json.dumps(probs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rule = report.get("rule")
    if rule is not None:
        log.info(
            f"  规则基线(t=0.5) 漏检={rule['false_pass_rate']:.1%} "
            f"误杀={rule['false_fail_rate']:.1%} 弃权={rule['cannot_verify_rate']:.1%}"
        )
    for point in report["points"]:
        fpr, ffr, cvr = (
            point["false_pass_rate"],
            point["false_fail_rate"],
            point["cannot_verify_rate"],
        )
        log.info(
            f"  t={point['threshold']:.2f} "
            f"漏检={'—' if fpr is None else f'{fpr:.1%}'} "
            f"误杀={'—' if ffr is None else f'{ffr:.1%}'} "
            f"弃权={'—' if cvr is None else f'{cvr:.1%}'}"
        )
    chosen = gate_sweep.operating_point(report)
    if chosen is None:
        log.error("no threshold reaches zero false passes: this gate needs the rule layer")
    else:
        log.info(
            f"推荐工作点：t={chosen['threshold']:.2f} "
            f"(漏检={chosen['false_pass_rate']:.1%}, 误杀={chosen['false_fail_rate']:.1%})"
        )
    log.info(f"report: {outdir / 'report.md'}")
    return EXIT_OK


def cmd_verify_eval(args: argparse.Namespace) -> int:
    log = console()
    corpus_path = Path(args.corpus) if args.corpus else verification_eval.default_corpus_path()
    if corpus_path.is_file():
        corpus = verification_eval.load_corpus(corpus_path)
        log.info(f"corpus: {corpus_path} ({len(corpus.get('samples', []))} samples)")
    else:
        days = verification_eval.available_days()
        if not days:
            log.error("no corpus and no captures to build one from")
            return EXIT_FAIL
        corpus = verification_eval.build_corpus(days)
        verification_eval.save_corpus(corpus_path, corpus)
        log.info(f"corpus: built {len(corpus['samples'])} samples -> {corpus_path}")

    wanted = [name.strip() for name in args.match.split(",") if name.strip()]
    matchers = {}
    if "legacy" in wanted:
        matchers["legacy"] = verification_eval.legacy_matcher
    if "typed" in wanted:
        matchers["typed"] = verification_eval.typed_matcher
    if "llm" in wanted:
        if not args.allow_llm:
            log.info("llm matcher skipped: pass --allow-llm to enable the paid baseline")
        else:
            agent_config = config_mod.load()
            provider = get_provider(agent_config)
            ok, detail = provider.available()
            if not ok:
                log.error(f"llm matcher unavailable: {detail}")
                return EXIT_FAIL
            # Cap the paid calls, and make every matcher answer the same subset.
            before = len(corpus.get("samples", []))
            corpus = verification_eval.subsample(corpus, args.llm_sample)
            log.info(
                f"llm baseline: judging {len(corpus['samples'])}/{before} samples "
                f"(all matchers scored on this subset)"
            )
            matchers["llm"] = verification_eval.llm_matcher(provider)
    if not matchers:
        log.error(f"no known matcher in {wanted!r}; expected legacy/typed/llm")
        return EXIT_FAIL

    report = verification_eval.compare_matchers(corpus, matchers)
    outdir = paths.runs_dir() / "verification" / f"{dt.datetime.now():%Y%m%d-%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    text = verification_eval.markdown(report)
    (outdir / "report.md").write_text(text, encoding="utf-8")
    (outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for entry in report["matchers"]:
        fpr = entry["false_pass_rate"]
        ffr = entry["false_fail_rate"]
        log.info(
            f"  - {entry['matcher']:<8} 漏检率={fpr:.1%} 误杀率={ffr:.1%} "
            f"({entry['false_pass_count']}/{entry['attacks']} attacks passed)"
            if fpr is not None and ffr is not None
            else f"  - {entry['matcher']:<8} 样本不足"
        )
    log.info(f"report: {outdir / 'report.md'}")

    baseline_path = verification_eval.default_baseline_path()
    if args.update_baseline:
        verification_eval.save_baseline(baseline_path, report)
        log.info(f"baseline updated: {baseline_path}")
    elif baseline_path.is_file():
        verdict = verification_eval.check_baseline(
            report, verification_eval.load_baseline(baseline_path)
        )
        if verdict.get("corpus_changed"):
            if args.corpus:
                log.info(
                    "baseline skipped: ad-hoc corpus "
                    f"({verdict.get('current_samples')} samples) is not comparable to "
                    f"the frozen baseline ({verdict.get('baseline_samples')} samples)"
                )
            else:
                log.error(
                    "baseline invalid: corpus changed "
                    f"({verdict.get('baseline_samples')} -> {verdict.get('current_samples')} samples); "
                    "re-freeze with ./agent verify eval --update-baseline"
                )
                return EXIT_FAIL
        elif verdict["compared_matchers"]:
            if verdict["ok"]:
                log.info(
                    "baseline: no regression vs "
                    + ", ".join(verdict["compared_matchers"])
                )
            else:
                for item in verdict["regressions"]:
                    log.error(
                        f"regression: {item['matcher']}.{item['metric']} "
                        f"{item['baseline']:.4f} -> {item['now']:.4f}"
                    )
                return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
