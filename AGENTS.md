# AGENTS.md — 本地 Agent Harness

这是「自己的 Agent」的工作区：harness 内核归本仓库，任务（负载）可陆续增加。

## 布局

- `agent` — 唯一入口（`./agent run|status|report|runs|annotate|backup|memory|experiment|verify|launchd|doctor`）
- `harness/` — 内核：runtime（幂等/锁/台账）、tools（权限与审计）、memory（文件优先）、
  decisions（类型化决策）、validators（验收标准）、verification_eval（测量闸门自身）、
  validator_probes（校验器不变式的对抗探针）、
  providers（模型适配）、loop（自研循环）、experiment（实验台）
- `tasks/<name>/task.toml` — 任务声明（步骤、工具白名单、可写路径、校验、预算）
- `memory/` — `runs/` 为自动写入的事实记录，`notes/` 为人工确认的长期结论
- `runs/` — 每次运行的产物、日志与 `runs.db` 台账（评测的唯一数据源）

## 改动规则

1. 新增能力走工具层（`harness/tools/`）并声明权限；不要在步骤脚本里裸调 subprocess。
2. 验收标准写进 `harness/validators.py` 并挂到 `task.toml`，不要靠提示词约束。
3. 任何会写外部状态的步骤都要：`--dry-run` 可跳过、失败可降级或可重试、结果写进台账。
4. 记忆分层：事实自动写 `memory/runs/`；结论必须人工确认后写 `memory/notes/`。
5. 不要修改 `daily-trends/tools/` 下既有脚本的接口；任务层只调用它们。
6. 步骤里的渲染/预览写入必须落在 `{run_dir}` 内。补丁第三方模块的常量时要检查它的派生量
   （`X = A / "b"` 这种在 import 期算出来的，改 `A` 不会改 `X`），否则 dry-run 会写穿到真实仓库。
7. 测试不要读会被运行改写的现场文件（`data/raw/` 会被 `fetch` 覆盖）；用 `tests/fixtures/` 下的冻结夹具。

## 验证

```bash
python3 -m unittest discover -t . -s tests -v
./agent doctor
./agent run daily-trends --date 2026-09-25 --compose replay --dry-run
./agent verify eval
./agent verify probes
./agent verify content
```

`-t .` 是必需的：测试用相对导入（`from .helpers import ...`），不指定顶层目录会直接
`ImportError`。冒烟日期用 2026-09-25——它是唯一三天校验全过的日期；2026-09-22 的 raw 抓取
曾被一次 dry-run 覆盖，其可核验率永久受损（见 `[policy]` 与 `runs.db`）。
