#!/bin/sh
# 独立 GPU SDK 与构建目录；编译和 CPU 契约测试不申请 GPU 设备。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${ROBOTTRACE_INFERENCE_OUTPUT:?Set a new direct child of bin/}"
: "${ROBOTTRACE_BUILD_LOG:?Set a new build log path}"
cd "$project"
output_relative=$(python3 -B - "$project" "$ROBOTTRACE_INFERENCE_OUTPUT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
target = Path(sys.argv[2])
if not target.is_absolute():
    target = root / target
print(target.relative_to(root))
PY
)
test ! -e "$ROBOTTRACE_BUILD_LOG"
mkdir -p "$(dirname -- "$ROBOTTRACE_BUILD_LOG")"
timeout --signal=INT --kill-after=5s 180s \
 python3 -B -m robottrace.inference_cuda container \
 --build-output "$ROBOTTRACE_INFERENCE_OUTPUT" -- \
 env "ROBOTTRACE_CUDA_OUTPUT=/work/$output_relative" bash -eu -c '
cmake -S ros2/robottrace_inference -B /tmp/robottrace-inference-cuda-build \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON \
 -DROBOTTRACE_CUDA_BUILD=ON -DORT_GPU_ROOT=/opt/rt/ort \
 -DCMAKE_INSTALL_PREFIX=/tmp/robottrace-inference-cuda-install
cmake --build /tmp/robottrace-inference-cuda-build --parallel 2
export LD_LIBRARY_PATH="/tmp/robottrace-inference-cuda-build:/opt/rt/ort/lib:/opt/rt/cudnn:/opt/rt/cuda:$LD_LIBRARY_PATH"
ctest --test-dir /tmp/robottrace-inference-cuda-build --verbose
cmake --install /tmp/robottrace-inference-cuda-build
cp -a /tmp/robottrace-inference-cuda-install/lib "$ROBOTTRACE_CUDA_OUTPUT/"
strip "$ROBOTTRACE_CUDA_OUTPUT/lib/robottrace_inference/inference_pipeline_cuda" "$ROBOTTRACE_CUDA_OUTPUT"/lib/*.so
export LD_LIBRARY_PATH="$ROBOTTRACE_CUDA_OUTPUT/lib:/opt/rt/ort/lib:/opt/rt/cudnn:/opt/rt/cuda:$LD_LIBRARY_PATH"
python3 -B -m robottrace.inference --record-build --provider cuda \
 --cuda-artifact-dir "$ROBOTTRACE_CUDA_OUTPUT"
ldd "$ROBOTTRACE_CUDA_OUTPUT/lib/robottrace_inference/inference_pipeline_cuda"
' > "$ROBOTTRACE_BUILD_LOG" 2>&1
