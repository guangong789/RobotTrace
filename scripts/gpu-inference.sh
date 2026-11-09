#!/bin/sh
# 仅为这个子进程指定 GPU 库、HOME 和缓存；不继承可能包含 CPU SDK 的库路径。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
gpu="$project/.deps/gpu"
ort="$gpu/onnxruntime-linux-x64-gpu-1.20.1"
cudnn="$gpu/cudnn-linux-x86_64-9.3.0.75_cuda12-archive"
[ "$#" -gt 0 ] || { echo 'Usage: gpu-inference.sh COMMAND [ARGS...]' >&2; exit 2; }
[ -z "${LD_PRELOAD:-}" ] || { echo 'Unexpected LD_PRELOAD; refusing GPU launch.' >&2; exit 1; }
[ "$(realpath "$gpu")" = "$gpu" ] || { echo 'GPU directory is not isolated.' >&2; exit 1; }
for path in "$ort/lib" "$cudnn/lib" "$gpu/home" "$gpu/cache/cuda" "$gpu/tmp"; do
    [ -d "$path" ] && [ "$(realpath "$path")" = "$path" ] || {
        echo "Missing or redirected isolated GPU path: $path" >&2
        exit 1
    }
done
cd "$project"
exec env HOME="$gpu/home" XDG_CACHE_HOME="$gpu/cache" \
    CUDA_CACHE_PATH="$gpu/cache/cuda" TMPDIR="$gpu/tmp" \
    LD_LIBRARY_PATH="$ort/lib:$cudnn/lib:/usr/local/cuda-12.4/lib64:/usr/lib/wsl/lib" \
    PYTHONDONTWRITEBYTECODE=1 "$@"
