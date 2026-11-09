#!/bin/sh
# 在原有 Humble 镜像中构建，复用挂载的 CPU SDK；不改镜像或 V1 二进制。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test -f "$project/.deps/onnxruntime-linux-x64-1.20.1/include/onnxruntime_cxx_api.h"
test -f "$project/models/mobilenet/data/expected-1.f32"
output_relative=bin/inference
if [ -n "${ROBOTTRACE_INFERENCE_OUTPUT:-}" ]; then
    output_relative=$(python3 -B - "$project" "$ROBOTTRACE_INFERENCE_OUTPUT" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve(strict=True)
requested = Path(sys.argv[2])
target = requested if requested.is_absolute() else root / requested
if (
    ".." in requested.parts
    or (root / "bin").is_symlink()
    or target.parent != root / "bin"
    or target.name == "inference"
    or target.exists()
    or target.is_symlink()
):
    raise SystemExit("Override requires a new direct child of bin/, excluding bin/inference")
subprocess.run(["git", "check-ignore", "--quiet", "--", str(target)], cwd=root, check=True)
print(target.relative_to(root))
PY
    )
fi
log_file=${ROBOTTRACE_BUILD_LOG:-"$project/evidence/inference-build.log"}
mkdir -p "$(dirname -- "$log_file")"
timeout --signal=INT --kill-after=5s 180s "$project/scripts/ros.sh" \
 env "ROBOTTRACE_INFERENCE_OUTPUT_REL=$output_relative" bash -eu -c '
cmake -S ros2/robottrace_inference -B /tmp/robottrace-inference-build \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/robottrace-inference-install
cmake --build /tmp/robottrace-inference-build --parallel 2
export LD_LIBRARY_PATH="/tmp/robottrace-inference-build:$LD_LIBRARY_PATH"
ctest --test-dir /tmp/robottrace-inference-build --verbose
cmake --install /tmp/robottrace-inference-build
target="$PWD/$ROBOTTRACE_INFERENCE_OUTPUT_REL"
if [ "$ROBOTTRACE_INFERENCE_OUTPUT_REL" != bin/inference ]; then
    test ! -e "$target" && test ! -L "$target"
fi
mkdir -p "$target"
cp -a /tmp/robottrace-inference-install/lib "$target/"
strip "$target/lib/robottrace_inference/inference_pipeline" "$target"/lib/*.so
export LD_LIBRARY_PATH="$target/lib:$PWD/.deps/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH"
if [ "$ROBOTTRACE_INFERENCE_OUTPUT_REL" = bin/inference ]; then
    python3 -B -m robottrace.inference --record-build
else
    python3 -B -c "
import sys
from pathlib import Path
from robottrace import inference
inference.BIN = Path(sys.argv[1])
inference.PROGRAM = inference.BIN / \"lib/robottrace_inference/inference_pipeline\"
inference.record_build()
" "$target"
fi
' > "$log_file" 2>&1
