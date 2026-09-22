#!/usr/bin/env bash
# Install (and optionally load) the launchd agent that runs a task unattended.
#
#   scripts/install_launchd.sh daily-trends          # 只写 plist，不启用
#   scripts/install_launchd.sh daily-trends --load   # 写入并交给 launchd（开始每天定时跑）
#
# 在你确认要下线 Codex 侧那条自动化之前，建议先只写 plist 不 --load，
# 否则同一时间会有两个进程写同一个站点仓库（harness 侧有锁，但没必要打架）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:-daily-trends}"
shift || true

cd "$HERE"
if [[ "${1:-}" == "--load" ]]; then
  ./agent launchd install "$TASK" --load
else
  ./agent launchd install "$TASK"
fi

./agent launchd status "$TASK"
