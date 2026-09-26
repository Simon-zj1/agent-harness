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
./agent verify corpus                                              # 从真实抓取构建标注语料
./agent verify eval                                                # 测量验收闸门自身的漏检率
```

## 结构

```
agent                     唯一入口
harness/                  内核
  runtime.py              幂等（task+date 只成功一次）、锁、预算、降级、通知
  ledger.py               runs.db：每次运行/步骤/工具调用/校验结果
  registry.py tools/      Tool 接口 + 权限模型（可写路径、命令白名单、网络、执行器）
  memory.py               文件优先记忆 + SQLite FTS5 索引（缺失时自动降级 LIKE）
  decisions.py            类型化决策：PASS / FAIL / CANNOT_VERIFY / ABSTAIN + 失败分类 + 证据
  validators.py           验收标准（结构 / 引用 / 防编造可核验率）
  verification_eval.py    测量验收闸门自身：标注语料 + 多 matcher 对照 + 漏检率/误杀率
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
可核验率、token、耗时与工具调用次数，并给出**成本—严格度 Pareto 前沿**（同一臂的多次运行取均值；
零成本的 replay 臂会被自动标注为基线而非可选项）。

已经跑过的实验不必重跑就能重画前沿：

```bash
./agent experiment compare daily-trends-compare
```

## 验收闸门自身也要被测量

校验器不是「写了就算数」的。`./agent verify eval` 用一份标注语料对照多个 matcher，
报告两个方向相反的指标：

语料分七类，由真实抓取自动生成：真实引用、可枚举漂移、**真实但未被引用**（应通过）；
前缀延伸、后缀伪造、**跨天取证**、完全无关（应不通过）。全部 matcher 评同一份样本
（LLM 裁判按 `--llm-sample` 抽样，降低成本）：

| matcher | 漏检率（放行编造来源） | 误杀率（拦下真实引用） | 判「无法核验」 | 成本 |
| --- | --- | --- | --- | --- |
| legacy（旧布尔闸门） | 50.0% | 0.0% | 0.0% | 免费 |
| typed（当前） | **0.0%** | **0.0%** | 28.6% | 免费 |
| llm（LLM-as-judge） | 0.0% | 5.3% | 0.0% | 1,605,770 tok / 175 次 |

分类别的判决（总漏检率会被易样本稀释，分类别不会）：

| matcher | 前缀延伸 | 后缀伪造 | 跨天取证 | 完全无关 | 真实引用 | 可枚举漂移 | 真实未被引用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | pass×25 | pass×25 | fail×25 | fail×25 | pass×25 | pass×25 | pass×25 |
| typed | **cannot_verify×25** | **cannot_verify×25** | fail×25 | fail×25 | pass×25 | pass×25 | pass×25 |
| llm | fail×25 | fail×25 | fail×25 | fail×25 | pass×25 | **fail×4**, pass×21 | pass×25 |

读法（这三行不是「谁赢」，是三种不同的代价）：

- **legacy 是真错**：共享前缀即判通过。它唯一能挡住的是"完全不在抓取里"的样本，
  所以把跨天取证这类它本来就会的题加进语料，反而把它的总漏检率从 66.7% 稀释到 50%。
- **typed 免费且零误杀**，代价是把两类前缀型攻击判成 `CANNOT_VERIFY` —— 而且**这些不确定性
  全部落在攻击样本上，在 6 天真实内容里一次都没触发**（每天 `cannot_verify=0`）。
  跨天取证它给的是明确 `fail`，不是含糊。
- **LLM 裁判既不漏检也不含糊，但会把合法引用判错**：4 条 `http://` / `?utm_source=` 这类
  等价漂移被拒，而它每次判决约 9.2k input token。它不支配 typed，typed 也不支配它。

**这个闸门能证明的天花板**：`plausible_uncited` 那一列——URL 确实在当天抓取里、文章却从未
引用它——三个 matcher 全判 PASS，而且**这是正确的**。它证明的是 provenance（来源确实被抓到过），
不是 support（来源支撑了那句话）。想证明 support 需要另一层，不在本闸门的能力范围内。

把当前数字冻结成基线，之后每次运行自动比对，漏检率或误杀率任一项回升就退出非零：

```bash
./agent verify eval --update-baseline    # 冻结
./agent verify eval                      # 比对，退化即失败
```

要自己复现这三行：

```bash
./agent verify eval --match legacy,typed,llm --allow-llm --llm-sample 25   # 会产生 API 费用
```

语料分五类，由真实抓取自动生成：真实引用、可枚举漂移（应通过）；前缀延伸、后缀伪造、
完全无关（应不通过）。`CANNOT_VERIFY` 既不算漏检也不算误杀，由任务声明的 `[policy]`
决定是否阻断（默认 fail-closed）。

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
