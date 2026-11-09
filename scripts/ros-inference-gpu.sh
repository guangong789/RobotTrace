#!/bin/sh
# 显式 GPU 入口；只给指定证据目录写权限，保留原有 ROS2 网络与 IPC。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${ROBOTTRACE_EVIDENCE_DIR:?Set a dedicated runtime evidence directory}"
cd "$project"
exec python3 -B -m robottrace.inference_cuda container \
    --evidence-dir "$ROBOTTRACE_EVIDENCE_DIR" -- "$@"
