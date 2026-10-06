# 每日技术趋势 · 自建 harness 版

这个任务把 `daily-trends/tools/RUNBOOK.md` 的流程从「在 Codex 里手动/托管跑」
搬到了自己的 harness 上：调度、记忆、验收、失败降级都归本仓库，
抓取与渲染仍然复用 `daily-trends/tools/` 里已有脚本（不改它们的命令行契约）。

## 跑一次

```bash
# 免费冒烟测试：复用当天已有内容，只生成预览，不碰站点仓库
./agent run daily-trends --date 2026-09-25 --compose replay --dry-run --skip-steps fetch

# 用 DeepSeek 重新撰写（分片，会消耗 token），仍然只生成预览
./agent run daily-trends --date 2026-09-25 --compose llm --dry-run

# 一次性整篇撰写（保留作为对照；实测会被输出上限截断，见下）
./agent run daily-trends --date 2026-09-25 --compose llm-single --dry-run

# 发布默认禁用：task.toml 把 hexo deploy 定为唯一写入者。
# 只有在明确接受“第二写入者”风险时才用 --publish。
# ./agent run daily-trends --publish

# 把整件事交给 Codex CLI（对照组，走实验台，仍然不发布）
./agent experiment run daily-trends-compare --date 2026-09-25 --allow-llm --execute
```

## 为什么要分片撰写

历史实测（2026-09-22 真实数据；当时使用 DeepSeek 旧模型 `deepseek-chat`）：

| 路线 | 输入 tokens | 输出 tokens | 结果 |
| --- | --- | --- | --- |
| 一次写完（全量 raw 上下文） | 133,892 | 16,384（撞上限） | JSON 截断，失败 |
| 一次写完（预筛上下文） | 36,340 | 16,384（撞上限） | JSON 截断，失败 |
| 分片（预筛上下文，6 次调用） | 72,438 | 13,945 | 20 条热点 + 10 条仓库，可核验率 1.00，校验全过 |

结论：一篇 20 条的双语稿件塞不进一次响应。`llm` 模式因此改成 map-reduce：
一次总览 + 四个热点分组 + 一次 GitHub，合并时把**分片内的引用编号换算成全局编号**，
丢弃无法回溯到当日抓取的引用，并强制 20/10 条上限（不依赖模型自觉）。

## 产物与契约

- `runs/<run_id>/content.json`：本次撰写的内容，所有校验都以它为准。
- `runs/<run_id>/site-preview/trends/<date>/index.html`：dry-run 下渲染的预览。
- `{tools_dir}/data/<date>.json`：正式内容存储；实验跑（trigger=experiment）不覆写它。
- `{site_repo}/trends/<date>/index.html`：正式发布产物（仅非 dry-run）。

## 校验（发布闸门）

1. `daily_trends_structure` — 双语字段、条目数量上限、分组完整性。
2. `daily_trends_references` — 引用编号可解析、URL 合法、无孤立引用。
3. `daily_trends_verifiable` — 正文引用的每条来源必须能在当日 raw 抓取结果里找到
   （防编造闸门；低于阈值即失败）。
4. `sitemap_sane` — 站点 sitemap 可解析、无重复 URL、不累积空行。
5. `daily_trends_brief` — 页首「今日速读」能按关注方向选出条目，每条有理由、有正文。
6. `daily_trends_depth` — 摘要是否写出了机制/证据与边界（**只报警**，用于跟踪阅读质量）。

任何必需校验失败 → 不发布，运行记为 failed 并通知。

## 今日速读（阅读侧）

页面顶部那 5 条由 `tools/brief.py` + `config/interests.json` 决定：

- 选：命中关注方向（Agent 工程 / 机器人具身 / AI 提效 / 模型技术动向）的优先，
  论文权重略高于新闻；栏目配额为洞见 3 / 论文 2 / 仓库 1。
- 过滤：融资估值、人事、政治政策、社会争议、名人口水战——**只匹配标题**，
  并且遇到方法词（MCP / harness / benchmark / 数据集 / 世界模型…）时不过滤。
- 呈现：每条固定四行——是什么 / 为什么推给你（命中的方向与关键词）/ 正文 / 边界。
  边界句由 `extract_limit` 从正文抽取，抽不到就写「正文未提及限制」，这是给撰写环节的信号。
- 正文完整保留：被速读过滤掉的内容仍然在页面下方，速读只做排序与筛选，不删内容。
（compose 阶段通过自身的结构/引用/可核验自检时会写入正式内容存储；
 如果后续 `sitemap_sane` 才失败，canonical 可能已经被 compose 更新。）
（实测过：模型多写 4 条热点时被上限校验挡下，运行失败而不是把超限内容发出去。）

## 降级策略

| 情况 | 处理 |
| --- | --- |
| 代理不可用 / 抓取失败 | 沿用已有 raw，标记 degraded，继续 |
| X 付费额度耗尽（402） | 如实写进 notes，继续 |
| git fetch/rebase 失败 | 回到同步前状态，用本地副本继续，标记 degraded |
| 渲染失败或产物缺失 | 标记 degraded，不发布 |
| 推送失败 | failed + 通知（产物已在本地） |

> 站点仓库走 SSH 别名（`git@github-simonzj1:...`）而不是 `origin`：本机网络会挡
> github.com 的 HTTPS，SSH 与 api.github.com 正常。换机器时改 `task.toml` 的 `[publish] remote` 即可。
