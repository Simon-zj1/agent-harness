# 本地自建 Agent Harness

项目介绍页（中英双语，含架构图与实测数据）：**https://www.simon-zj.top/tech/agent/local-agent-harness/**

一套**自己的**本地 Agent 运行时：调度、记忆、权限、验收都归这台机器和自己，
模型与重活外包给可替换的执行器（DeepSeek API / Codex CLI / Claude Code）。

核心判断：RAG 不是「自己」的来源，它只是上下文装填手段。真正属于你的是四件事——
**记忆归你、权限归你、触发归你、工件与验收标准归你**。本仓库把这四项做进内核。

## 一分钟上手

```bash
cd /path/to/agent-harness
./agent doctor                                                     # 环境自检
./agent tasks                                                      # 已声明的任务
./agent run daily-trends --date 2026-09-25 --compose replay --dry-run   # 免费冒烟
./agent status                                                     # 台账 / 记忆 / provider
./agent report <run_id>                                            # 单次运行详情
./agent verify corpus                                              # 从真实抓取构建标注语料
./agent verify eval                                                # 测量验收闸门自身的漏检率
```

`verify corpus` / `verify eval` 依赖仓库外的 `daily-trends` 抓取数据。默认取本仓库
同级的 `../daily-trends/data`，也可以用 `DAILY_TRENDS_DATA_DIR` 显式覆盖；`doctor`
会检查任务声明的外部读取路径是否存在。

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
  loop.py                 实验性自研 agent loop（工具调用 + 预算 + 停止条件；不用于 daily-trends 主路径）
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

`loop.py` 里的自研 loop 有独立测试，但当前 `daily-trends` 的 `llm` 路径走的是
`providers` 直接调用的分片 map-reduce，而不是它；两者不互相替代，也不混称生产主循环。

## 与「直接用 Codex 读本地文件」的区别

| 维度 | 在 Codex 里读本地文件 | 本 harness |
| --- | --- | --- |
| 会话与记忆 | 会话在厂商的存储里，跨会话记忆弱 | `memory/` 全在本机，可 git、可审计、可删 |
| 触发 | 你打开它才会跑 | launchd 定时/事件触发，无人值守 |
| 权限 | 围绕工作目录 | 外部写入声明可写路径 + 命令白名单；经 registry 的工具调用全部入台账 |
| 验收 | 靠你读输出 | 校验器代码化，可核验率不达标就不发布 |
| 模型 | 绑定某个 harness | provider 可换（DeepSeek/本地/其它）；自研 loop 目前是实验模块 |

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

当前作者机器上的 `com.simonzj.agent.daily-trends` 并未加载；上面的命令是安装/启用步骤，
不是“已经在无人值守运行”的证据。用 `./agent launchd status daily-trends` 查看真实状态。

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
| legacy（旧布尔闸门） | 62.1% | 0.6% | 0.0% | 免费 |
| typed（当前） | **0.0%** | **0.0%** | 32.3% | 免费 |

LLM 裁判只在 175 条子样本上运行过（1,605,770 input token / 175 次），不能和上面的
930 条主表直接比较；它是历史对照，不是当前默认基线。

分类别的判决（总漏检率会被易样本稀释，分类别不会）：

读法（这不是「谁赢」，是三种不同的代价）：

- **legacy 是真错**：共享前缀即判通过。它唯一能挡住的是"完全不在抓取里"的样本；
  7 类语料合在一起时，它在 560 条攻击上放行了 348 条。
- **typed 免费且零误杀**，代价是把两类前缀型攻击判成 `CANNOT_VERIFY`；这些不确定性
  落在攻击样本上，而不是合法引用上。
- **LLM 裁判在 175 条子样本上不漏检，但会把合法引用判错**：4 条 `http://` /
  `?utm_source=` 这类等价漂移被拒，而它每次判决约 9.2k input token。它不支配 typed，
  typed 也不支配它。

**这个闸门能证明的天花板**：`plausible_uncited` 那一列——URL 确实在当天抓取里、文章却从未
引用它——三个 matcher 全判 PASS，而且**这是正确的**。它证明的是 provenance（来源确实被抓到过），
不是 support（来源支撑了那句话）。想证明 support 需要另一层，不在本闸门的能力范围内。

把当前数字冻结成基线，之后每次运行自动比对，漏检率或误杀率任一项回升就退出非零：

```bash
./agent verify eval --update-baseline    # 冻结
./agent verify eval                      # 比对，退化即失败
```

如果默认语料的组成发生变化，`verify eval` 会拒绝拿旧基线做比较并退出非零，要求
重新冻结；显式传入 `--corpus` 的临时评估会跳过基线，因为它本来就不是同一份语料。

### 校验器对抗探针

有些不变式没有可生成的语料。退款闸门的规则是「查不到的前提一律不得批准」，
测它的方式是拿违反该规则的载荷去撞它：

```bash
./agent verify probes
```

`refund_decisions_fail_closed` 的 10 个探针全部通过，其中 3 个是**应该放行**的用例
（含"规则不满足因此拒绝"这种正确行为）——只会拒绝的校验器同样拿不到分。
测试里用「永远放行」和「永远拒绝」两个假校验器做了反向对照，确认探针能识别出坏闸门，
否则探针只是装饰。

### 被拦下之后：修复简报

闸门只说「不行」等于把问题丢回给人。类型化结果里本来就带着证据和修复建议，
所以被拦下的运行会写出 `runs/<run_id>/validation-failures.json`，`agent report`
也会直接打印：

```
$ ./agent report 2026-09-22-daily-trends-141800-ff91b7
被拦下：1 个校验器未通过
  · daily_trends_verifiable  decision=fail class=fabricated_source
    28/47 cited references trace back to the raw capture (cannot_verify=0, fail=19)
      - ref#1 https://arxiv.org/abs/2609.24974v1  no fetched url matches
      - ref#2 https://arxiv.org/abs/2609.24972v1  no fetched url matches
      … 另有 14 条，见 validation-failures.json
    修复建议：Re-fetch the cited page, or replace the citation with a URL that is present in the raw capture.
```

`decision`、`failure_class`、逐条 `evidence`（哪一条引用、哪个 URL、为什么）都会
写进台账并持久化，所以一次失败是可复查、可交接、可被后续修复步骤消费的，而不是
一句 exit 1。

### 历史内容债

闸门在运行当时拦截。但在此之前的每一天已经发出去了，那些内容按现在的标准是过不了的。
把它们一次看清，而不是靠一次失败的运行去发现：

```bash
./agent verify content
```

```
  FAIL 2026-09-20  ratio=0.8806 fail=8  orphans=1 blockers=references,verifiable
  ok   2026-09-21  ratio=1.0    fail=0  orphans=0 blockers=-
  FAIL 2026-09-22  ratio=0.5957 fail=19 orphans=0 blockers=verifiable
  FAIL 2026-09-23  ratio=1.0    fail=0  orphans=2 blockers=references
  FAIL 2026-09-24  ratio=1.0    fail=0  orphans=2 blockers=references
  ok   2026-09-25  ratio=1.0    fail=0  orphans=0 blockers=-
  ok   2026-09-26  ratio=1.0    fail=0  orphans=0 blockers=-
  FAIL 2026-09-27  ratio=1.0    fail=0  orphans=0 blockers=structure
干净 3/8 天
```

`ratio` 掉下来有两种成因，报告里要求分开看：引用确实不在当天抓取里，或者那天的抓取
文件被后续运行覆盖过（`fetch` 曾在 dry-run 下也执行）。后者是可复现性事故，不是内容问题。
2026-09-27 则是结构问题：canonical 文件里 20 条热点缺少双语 summary/comment，
共 60 条结构错误；`verify content` 已把它标为 `blockers=structure`。

这个命令**只诊断，不改已发布内容**——是否回修历史稿件是编辑决定。

要自己复现这三行：

```bash
./agent verify eval --match legacy,typed,llm --allow-llm --llm-sample 25   # 会产生 API 费用
```

语料分七类，由真实抓取自动生成：真实引用、可枚举漂移、真实但未被引用（应通过）；
前缀延伸、后缀伪造、跨天取证、完全无关（应不通过）。`CANNOT_VERIFY` 既不算漏检也不算
误杀，由任务声明的 `[policy]` 决定是否阻断（默认 fail-closed）。

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
`budget.monthly_limit_usd` 是月闸门，`budget.max_cost_per_run_usd` 是单次 compose
的成本闸门；超过单次上限时不会写入正式内容存储，也不会继续渲染/发布。
如果启用 webhook 通知，`notify.webhook_allow_hosts` 是目标主机白名单；非空时只允许
列表内主机。

每次真实运行结束都会生成 `runs/runs.db.backup`；需要把台账和记忆一起快照到指定目录：

```bash
./agent backup --out /path/to/backup-dir
```

任务里的本机路径默认指向仓库同级目录，或写成 `${VAR:-默认值}`，换机器不用改文件：

```bash
export DAILY_TRENDS_DIR=/path/to/daily-trends
export SITE_REPO_DIR=/path/to/your-site
```

`pr-guard` 需要调用者声明变更范围，可以用可重复的 `--env` 传入：

```bash
./agent run pr-guard \
  --env AGENT_PR_RANGE=HEAD~1..HEAD \
  --env AGENT_PR_SCOPE=harness/,tasks/,tests/,config/,experiments/,scripts/,.github/
```

抓取与再分发的合规边界记录在 [COMPLIANCE.md](COMPLIANCE.md)；其中 X 回退来源和
第三方站点条款标为“需外部确认”，不是已完成的法律结论。
