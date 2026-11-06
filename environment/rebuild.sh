#!/bin/sh
# 默认复用镜像；冷重建需显式触发，设有时限，并使用项目内配置。
set -eu
config=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
force=0
proxy_download=0
for arg in "$@"; do
    case "$arg" in
        --force) force=1 ;;
        --proxy-download) proxy_download=1 ;;
        *) echo 'Usage: environment/rebuild.sh [--force] [--proxy-download]' >&2; exit 2 ;;
    esac
done
if [ "$force" = 0 ] && "$config/image.sh" >/dev/null 2>&1; then
    echo 'Recorded local runtime available; no rebuild required.'
    exit 0
fi
runtime=$(cat "$config/image-reference.txt")
base_digest=$(cat "$config/base-image-digest.txt")
imported=$(cat "$config/imported-base-image-id.txt")
base="ros@$base_digest"
if timeout 10s docker image inspect "$imported" >/dev/null 2>&1; then
    base=robottrace/humble-base:r0-00
    docker tag "$imported" "$base"
elif [ "$proxy_download" = 1 ]; then
    work_dir=$(mktemp -d "${TMPDIR:-/tmp}/robottrace-image.XXXXXX")
    trap 'rm -rf -- "$work_dir"' EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    curl --fail --location --retry 0 --connect-timeout 10 --max-time 120 \
        https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz \
        --output "$work_dir/crane.tar.gz"
    echo "0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0  $work_dir/crane.tar.gz" | sha256sum --check
    tar -xzf "$work_dir/crane.tar.gz" -C "$work_dir" crane
    timeout --signal=TERM --kill-after=10s 360s "$work_dir/crane" pull --platform linux/amd64 "ros@$base_digest" "$work_dir/base.tar"
    # 读取固定版本 crane 生成的 Docker 归档中记录的准确标签。
    loaded_tag=$(python3 - "$work_dir/base.tar" <<'PY'
import json, sys, tarfile

with tarfile.open(sys.argv[1]) as archive:
    manifest = json.load(archive.extractfile("manifest.json"))
    assert len(manifest) == 1 and len(manifest[0]["RepoTags"]) == 1
    print(manifest[0]["RepoTags"][0])
PY
    )
    timeout --signal=TERM --kill-after=5s 120s docker load --input "$work_dir/base.tar"
    base=$loaded_tag
else
    timeout --signal=TERM --kill-after=10s 240s docker pull --platform linux/amd64 "$base" || {
        echo 'Pinned-base pull failed. Investigate the daemon network, or retry explicitly with --proxy-download using the existing shell proxy.' >&2
        exit 1
    }
fi
timeout --signal=TERM --kill-after=10s 360s docker build --network host --platform linux/amd64 \
    --pull=false --progress plain --build-arg "ROS_BASE=$base" \
    --build-arg http_proxy --build-arg https_proxy --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
    --tag "$runtime" "$config"
# 这里记录所选构建的镜像身份；是否验证通过应由测试确认，而不是文件名。
rebuilt_id=$(timeout 10s docker image inspect "$runtime" --format '{{.Id}}')
printf '%s\n' "$rebuilt_id" > "$config/image-id.txt"
echo 'Image reconstructed. Run scripts/build.sh and the documented tests before relying on it.'
