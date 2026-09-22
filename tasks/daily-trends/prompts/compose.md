请为 {DATE}（北京时间 00:00–23:00 口径）撰写「每日技术趋势」内容。

## 输出 JSON 结构（严格遵守，键名不可更改）

date 填 {DATE}，generated_at 填本地时间，sections 必须是 insights 与 github 两节。

- insights 节：title / intro / groups；四个 group 的 id 依次为
  `agent-engineering`、`robotics`、`ai-productivity`、`industry-moves`，每个 group 有
  title 与 items。
- github 节：title / intro / items（无 groups）。
- insights 与 github 的每个 item 都有：title{zh,en}、summary{zh,en}、comment{zh,en}、sources[整数]；
  github 的 item 额外有 meta{stars,stars_per_day,language}。
- 顶层还要有：title{zh,en}、summary{zh,en}、tldr{zh[],en[]}（各 5 条）、
  references[{id,title,url,source,date}]、notes{zh,en}、stats{insights,repos}。

## 写作要求

- summary 陈述事实并在关键数字/技术名后用 [[n]] 标注引用；comment 是你的判断，不复述事实。
- references 只收录正文确实引用过的来源，url 必须来自证据中的 URL 全集。
- 证据不足时宁可少写；某类证据为空时在 notes 里如实说明数据源降级。
- 中文里技术缩写首次出现要给出英文全称或简短解释。

{MEMORY}

## 证据（当日真实抓取结果）

{EVIDENCE}

只输出 JSON 对象，不要代码围栏，不要任何解释文字。
