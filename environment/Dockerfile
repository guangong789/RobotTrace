# 使用公开且固定版本的输入重建原有 Humble 运行环境。
# ROS_BASE 可指定已经导入并核验过来源的本地基础镜像。
ARG ROS_BASE=ros@sha256:a3f4de78e42cca693a543ece5798aedcf1cae0171fe71c334a047332cc72ef93
FROM ${ROS_BASE}
RUN sed -i 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g; s|http://security.ubuntu.com|https://security.ubuntu.com|g' /etc/apt/sources.list \
 && apt-get -o APT::Update::Error-Mode=any -o Acquire::Retries=0 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
 && DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
      ros-humble-demo-nodes-cpp=0.20.10-1jammy.20260907.223640 \
      ros-humble-demo-nodes-py=0.20.10-1jammy.20260907.214920 \
      ros-humble-ros2bag=0.15.17-1jammy.20260908.095407 \
      ros-humble-rosbag2-transport=0.15.17-1jammy.20260908.093213 \
      ros-humble-rosbag2-storage-default-plugins=0.15.17-1jammy.20260908.084722 \
      ros-humble-rosbag2-storage-mcap=0.15.17-1jammy.20260908.085225 \
 && rm -rf /var/lib/apt/lists/*
LABEL org.robottrace.environment="Humble runtime reconstructed from pinned R0 provenance"
