"""独立于 ROS 的事件关联；导入本包不会启动运行环境。"""

from .events import Event, RunEvidence, CollectorEvidence
from .analysis import analyze

__all__ = ["Event", "RunEvidence", "CollectorEvidence", "analyze"]
