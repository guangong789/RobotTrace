#!/bin/sh
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
log_file=${ROBOTTRACE_BUILD_LOG:-"$project/evidence/build.log"}
mkdir -p "$(dirname -- "$log_file")"
timeout --signal=INT --kill-after=5s 150s "$project/scripts/ros.sh" bash -eu -c '
colcon --log-base /tmp/robottrace-log build --base-paths ros2 --build-base /tmp/robottrace-build \
 --install-base /tmp/robottrace-install --packages-select robottrace_examples \
 --cmake-args -DCMAKE_BUILD_TYPE=Release --event-handlers console_direct+
mkdir -p bin
cp /tmp/robottrace-install/robottrace_examples/lib/robottrace_examples/pipeline bin/
cp /tmp/robottrace-install/robottrace_examples/lib/robottrace_examples/event_cost bin/
strip bin/pipeline bin/event_cost
' > "$log_file" 2>&1
