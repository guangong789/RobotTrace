#!/bin/sh
# 只构建被选中的上游源码；临时构建树随容器销毁。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
log_file=${ROBOTTRACE_BUILD_LOG:-"$project/evidence/upstream-build.log"}
mkdir -p "$(dirname -- "$log_file")"
timeout --signal=INT --kill-after=5s 180s "$project/scripts/ros.sh" bash -eu -c '
cmake -S ros2/robottrace_upstream -B /tmp/robottrace-upstream-build \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/robottrace-upstream-install
cmake --build /tmp/robottrace-upstream-build --parallel 2
/tmp/robottrace-upstream-build/identity_contract
cmake --install /tmp/robottrace-upstream-build
mkdir -p bin/upstream
cp /tmp/robottrace-upstream-install/lib/robottrace_upstream/talker bin/upstream/
cp /tmp/robottrace-upstream-install/lib/robottrace_upstream/listener bin/upstream/
cp /tmp/robottrace-upstream-install/lib/robottrace_upstream/relay bin/upstream/
cp /tmp/robottrace-upstream-install/lib/robottrace_upstream/librobottrace_demo_topics.so bin/upstream/
cp /tmp/robottrace-upstream-install/lib/robottrace_upstream/librobottrace_relay.so bin/upstream/
strip bin/upstream/talker bin/upstream/listener bin/upstream/relay bin/upstream/*.so
export LD_LIBRARY_PATH="$PWD/bin/upstream:$LD_LIBRARY_PATH"
python3 -B -m robottrace.upstream --record-build
' > "$log_file" 2>&1
