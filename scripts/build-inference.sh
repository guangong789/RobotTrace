#!/bin/sh
# 在原有 Humble 镜像中构建，复用挂载的 CPU SDK；不改镜像或 V1 二进制。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test -f "$project/.deps/onnxruntime-linux-x64-1.20.1/include/onnxruntime_cxx_api.h"
test -f "$project/models/mobilenet/data/expected-1.f32"
log_file=${ROBOTTRACE_BUILD_LOG:-"$project/evidence/inference-build.log"}
mkdir -p "$(dirname -- "$log_file")"
timeout --signal=INT --kill-after=5s 180s "$project/scripts/ros.sh" bash -eu -c '
cmake -S ros2/robottrace_inference -B /tmp/robottrace-inference-build \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/robottrace-inference-install
cmake --build /tmp/robottrace-inference-build --parallel 2
export LD_LIBRARY_PATH="/tmp/robottrace-inference-build:$LD_LIBRARY_PATH"
ctest --test-dir /tmp/robottrace-inference-build --verbose
cmake --install /tmp/robottrace-inference-build
mkdir -p bin/inference
cp -a /tmp/robottrace-inference-install/lib bin/inference/
strip bin/inference/lib/robottrace_inference/inference_pipeline bin/inference/lib/*.so
export LD_LIBRARY_PATH="$PWD/bin/inference/lib:$PWD/.deps/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH"
python3 -B -m robottrace.inference --record-build
' > "$log_file" 2>&1
