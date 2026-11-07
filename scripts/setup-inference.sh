#!/bin/sh
# 仅准备这个实验的固定 CPU SDK 和模型；不安装软件或更改 V1 镜像。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sdk="$project/.deps/onnxruntime-linux-x64-1.20.1"
[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || {
    echo 'This experiment requires Linux x86_64.' >&2
    exit 1
}
mkdir -p "$project/.deps"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/robottrace-inference-setup.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [ ! -d "$sdk" ]; then
    curl --fail --location --connect-timeout 10 --max-time 120 \
        'https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz' \
        --output "$scratch/ort.tgz"
    printf '%s  %s\n' '67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105' "$scratch/ort.tgz" | sha256sum --check
    tar -xzf "$scratch/ort.tgz" -C "$project/.deps"
fi
[ "$(cat "$sdk/VERSION_NUMBER")" = 1.20.1 ]
vision_data="$project/models/mobilenet/data"
mkdir -p "$vision_data"
if [ ! -f "$vision_data/mobilenetv2-12.onnx" ]; then
    curl --fail --location --connect-timeout 10 --max-time 120 \
        'https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/23a2a25eaaa6fcd61865414c8a1acf13dda66b7a/mobilenetv2-12.onnx' \
        --output "$scratch/mobilenet.onnx"
    printf '%s  %s\n' 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5' "$scratch/mobilenet.onnx" | sha256sum --check
    mv "$scratch/mobilenet.onnx" "$vision_data/mobilenetv2-12.onnx"
fi
if [ ! -f "$vision_data/dog.jpg" ]; then
    curl --fail --location --connect-timeout 10 --max-time 60 \
        'https://raw.githubusercontent.com/pytorch/hub/c3beaae7d32fca2a23fec30aa7938ef5c9b6e5d5/images/dog.jpg' \
        --output "$scratch/dog.jpg"
    printf '%s  %s\n' 'f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a' "$scratch/dog.jpg" | sha256sum --check
    mv "$scratch/dog.jpg" "$vision_data/dog.jpg"
fi
printf '%s  %s\n' 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5' "$vision_data/mobilenetv2-12.onnx" | sha256sum --check
printf '%s  %s\n' 'f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a' "$vision_data/dog.jpg" | sha256sum --check
echo 'CPU SDK and models ready. Generate vision fixtures once as described in models/mobilenet/README.md.'
