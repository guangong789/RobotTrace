#!/bin/sh
# 仅解析本地镜像；启动时不会附带拉取或重建镜像。
set -eu
config=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
default_image=$(cat "$config/image-reference.txt")
expected=$(cat "$config/image-id.txt")
image=${ROBOTTRACE_IMAGE:-$default_image}
if actual=$(timeout 10s docker image inspect "$image" --format '{{.Id}}' 2>/dev/null); then
    if [ -z "${ROBOTTRACE_IMAGE:-}" ] && [ "$actual" != "$expected" ]; then
        echo 'Project image tag differs from the recorded image ID; refusing an implicit replacement.' >&2
        exit 1
    fi
elif [ -z "${ROBOTTRACE_IMAGE:-}" ] && timeout 10s docker image inspect "$expected" >/dev/null 2>&1; then
    docker tag "$expected" "$image"
else
    echo 'Local ROS2 image unavailable. Check Docker, then run environment/rebuild.sh; use --proxy-download if required.' >&2
    exit 1
fi
printf '%s\n' "$image"
