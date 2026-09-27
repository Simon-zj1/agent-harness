# PR 合并守卫

这个任务把一次变更拆成三个独立判决：

- `TESTS_PASS`：测试套件是否真的运行并有汇总；
- `ARCHITECTURE_OK`：仓库在 AGENTS.md 里声明的分层规则是否成立；
- `NO_UNINTENDED_SCOPE`：变更文件是否都在调用者声明的范围内。

`AGENT_PR_SCOPE` 没有默认值。没有 scope 时，`NO_UNINTENDED_SCOPE` 会返回
`cannot_verify`，最终 `merge=block`；这是 fail-closed，不是 bug。

运行示例：

```bash
./agent run pr-guard \
  --env AGENT_PR_RANGE=HEAD~1..HEAD \
  --env AGENT_PR_SCOPE=harness/,tasks/,tests/,config/,experiments/,scripts/,.github/
```

验收层 `pr_merge_gate` 要求三条 required checks 恰好各出现一次、decision 合法，
并且 artifact 明确声明 `merge=allow`，否则不自动合并。
