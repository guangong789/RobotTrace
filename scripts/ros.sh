#!/bin/sh
# 使用项目内配置选择运行镜像、挂载目录和宿主机 UID/GID；此处不重建镜像。
set -eu
project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
image=$("$project/environment/image.sh")
if [ "$#" -eq 0 ]; then set -- bash --noprofile --norc -i; fi
if [ -t 0 ] && [ -t 1 ]; then terminal=-it; else terminal=-i; fi
set -- --entrypoint /bin/bash "$image" -c \
  'mkdir -p "$HOME"; source /opt/ros/humble/setup.bash; exec "$@"' robottrace "$@"
# 隔离验证输出，不修改应用、测试代码或历史证据。
if [ -n "${ROBOTTRACE_EVIDENCE_DIR:-}" ]; then
    evidence=$(python3 -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]).resolve(strict=True); root=Path(sys.argv[2]).resolve(strict=True); assert p.is_dir() and p!=root and p.is_relative_to(root); print(p)' "$ROBOTTRACE_EVIDENCE_DIR" "$project")
    set -- --volume "$evidence:/work/evidence" "$@"
fi
exec docker run --rm --init "$terminal" --label robottrace=r1-01 \
  --network host --ipc host --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/robottrace-home --env ROS_DOMAIN_ID=42 --env ROS_LOCALHOST_ONLY=1 \
  --env RMW_IMPLEMENTATION=rmw_fastrtps_cpp --env RCUTILS_COLORIZED_OUTPUT=0 \
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --volume "$project:/work" --workdir /work "$@"
