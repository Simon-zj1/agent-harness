"""Command line interface: run / status / report / memory / experiment / launchd / doctor."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

from . import __version__, config as config_mod, launchd, memory, paths, taskspec
from .errors import AlreadyDone, ConfigError, HarnessError, LockBusy
from .experiment import load as load_experiment, run_experiment
from .ledger import Ledger
from .logutil import console
from .providers import get as get_provider
from .runtime import RunOptions, Runner, default_date

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
    return parser


# -- commands --------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    runner = Runner()
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

    ledger = Ledger()
    log.info(f"ledger      : {ledger.path}")
    log.info(f"now         : {dt.datetime.now().astimezone().isoformat(timespec='seconds')}")

    if problems:
        log.error("problems:")
        for problem in problems:
            log.error(f"  - {problem}")
        return EXIT_FAIL
    log.info("all checks passed")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
