# 本地自建 Agent Harness

项目介绍页（中英双语，含架构图与实测数据）：**https://www.simon-zj.top/tech/agent/local-agent-harness/**

一套**自己的**本地 Agent 运行时：调度、记忆、权限、验收都归这台机器和自己，
模型与重活外包给可替换的执行器（DeepSeek API / Codex CLI / Claude Code）。

核心判断：RAG 不是「自己」的来源，它只是上下文装填手段。真正属于你的是四件事——
**记忆归你、权限归你、触发归你、工件与验收标准归你**。本仓库把这四项做进内核。

## 一分钟上手

```bash
cd /Users/simon-zj/Documents/ChatGPT/Agent
./agent doctor                                                     # 环境自检
./agent tasks                                                      # 已声明的任务
./agent run daily-trends --date 2026-09-22 --compose replay --dry-run   # 免费冒烟
./agent status                                                     # 台账 / 记忆 / provider
./agent report <run_id>                                            # 单次运行详情
```

## 结构

```
agent                     唯一入口
harness/                  内核
  runtime.py              幂等（task+date 只成功一次）、锁、预算、降级、通知
  ledger.py               runs.db：每次运行/步骤/工具调用/校验结果
  registry.py tools/      Tool 接口 + 权限模型（可写路径、命令白名单、网络、执行器）
  memory.py               文件优先记忆 + SQLite FTS5 索引（缺失时自动降级 LIKE）
  validators.py           验收标准（结构 / 引用 / 防编造可核验率）
  providers/              模型适配层（DeepSeek 默认；本地 OpenAI 兼容端点预留）
  loop.py                 自研 agent loop（工具调用 + 预算 + 停止条件）
  experiment.py           多策略对比实验台
  launchd.py              从 task.toml 生成无人值守触发器
tasks/daily-trends/       首个真实负载：把现有 RUNBOOK 流程搬进 harness
experiments/              对比实验定义（自建 vs 托管、上下文/记忆策略 A/B）
memory/ runs/             记忆与运行产物（AGENT_HOME 可整体迁移）
```

`--compose` 三种模式：`replay`（复用当天的内容，免费、确定性）、
`llm`（分片撰写：一次总览 + 四个分组 + 一次 GitHub，合并时统一引用编号并强制 20/10 上限）、
`llm-single`（整篇一次写完，保留作对照——实测会被输出上限截断）、
`delegate`（整件事交给 Codex/Claude）。

## 与「直接用 Codex 读本地文件」的区别

| 维度 | 在 Codex 里读本地文件 | 本 harness |
| --- | --- | --- |
| 会话与记忆 | 会话在厂商的存储里，跨会话记忆弱 | `memory/` 全在本机，可 git、可审计、可删 |
| 触发 | 你打开它才会跑 | launchd 定时/事件触发，无人值守 |
| 权限 | 围绕工作目录 | 每个任务声明可写路径 + 命令白名单，工具调用全部入台账 |
| 验收 | 靠你读输出 | 校验器代码化，可核验率不达标就不发布 |
| 模型 | 绑定某个 harness | provider 可换（DeepSeek/本地/其它），loop 是自己的 |

如果需求只是「问答 + 读本地文件」，用 Codex 更划算；本 harness 值得存在的前提是
**无人值守 + 跨月累积的记忆 + 固定工件与验收标准**。

## 无人值守调度

```bash
./agent launchd render daily-trends --out /tmp/daily-trends.plist   # 先看生成物
scripts/install_launchd.sh daily-trends                             # 写入 ~/Library/LaunchAgents（不启用）
scripts/install_launchd.sh daily-trends --load                      # 交给 launchd 定时执行
```

`--load` 之前请确认 Codex 侧那条同任务的自动化已暂停：两条都跑会重复发布
（harness 侧有站点仓库锁，但没必要让它们打架）。

## 实验台

```bash
./agent experiment list
./agent experiment run daily-trends-compare --date 2026-09-22                    # 只跑离线臂
./agent experiment run daily-trends-compare --date 2026-09-22 --allow-llm        # 加上 DeepSeek 臂
./agent experiment run daily-trends-compare --date 2026-09-22 --allow-llm --arm harness-llm-prefilter
./agent experiment run daily-trends-compare --date 2026-09-22 --allow-llm --arm codex-exec --execute
```

臂默认跳过 `sync-site`/`fetch`/`publish`：所有臂跑同一份已抓取的 raw，比较才公平，
也不会碰站点仓库。`--arm` 可以只跑其中几条；`--execute` 让臂真正执行（dry-run 不会启动执行器）。

报告落在 `runs/experiments/<experiment_id>/report.{md,json,csv}`，对比成功率、校验通过、
可核验率、token、耗时与工具调用次数。

## 记忆

```bash
./agent memory search "上下文工程"
./agent memory add --title "harness 设计取舍" --body "..." --tags agent,harness
./agent memory promote <run_id> --title "2026-09-22 发布复盘"
```

`memory/runs/` 由运行自动追加（事实）；`memory/notes/` 只在明确要求时写入（结论）。

## 配置

`config/agent.toml`：默认 provider/executor、预算、通知、命令白名单、执行器参数。
密钥从环境变量读取，回退 `~/.codex/.env`；本仓库任何文件都不存明文密钥。

任务里的本机路径写成 `${VAR:-默认值}`，换机器不用改文件：

```bash
export DAILY_TRENDS_DIR=/path/to/daily-trends
export SITE_REPO_DIR=/path/to/your-site
export BLOG_REPO_DIR=/path/to/hexo-source
```
