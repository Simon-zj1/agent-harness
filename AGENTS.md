# AGENTS.md — 本地 Agent Harness

这是「自己的 Agent」的工作区：harness 内核归本仓库，任务（负载）可陆续增加。

## 布局

- `agent` — 唯一入口（`./agent run|status|report|runs|annotate|memory|experiment|launchd|doctor`）
- `harness/` — 内核：runtime（幂等/锁/台账）、tools（权限与审计）、memory（文件优先）、
  validators（验收标准）、providers（模型适配）、loop（自研循环）、experiment（实验台）
- `tasks/<name>/task.toml` — 任务声明（步骤、工具白名单、可写路径、校验、预算）
- `memory/` — `runs/` 为自动写入的事实记录，`notes/` 为人工确认的长期结论
- `runs/` — 每次运行的产物、日志与 `runs.db` 台账（评测的唯一数据源）

## 改动规则

1. 新增能力走工具层（`harness/tools/`）并声明权限；不要在步骤脚本里裸调 subprocess。
2. 验收标准写进 `harness/validators.py` 并挂到 `task.toml`，不要靠提示词约束。
3. 任何会写外部状态的步骤都要：`--dry-run` 可跳过、失败可降级或可重试、结果写进台账。
4. 记忆分层：事实自动写 `memory/runs/`；结论必须人工确认后写 `memory/notes/`。
5. 不要修改 `daily-trends/tools/` 下既有脚本的接口；任务层只调用它们。

## 验证

```bash
python3 -m unittest discover -s tests -v
./agent doctor
./agent run daily-trends --date 2026-09-22 --compose replay --dry-run
```
