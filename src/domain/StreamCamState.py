import threading
import time
from typing import Optional

import numpy as np


class StreamCamState:
    """
    本地的 StreamState：
    - latest_frame: 最新一帧的原始 BGR 图像（OpenCV 格式）
    - latest_ts   : 最新一帧的时间戳
    """

    def __init__(self):
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_ts: float = 0.0
        self._lock = threading.Lock()

    def update_frame(self, frame: np.ndarray):
        with self._lock:
            self.latest_frame = frame
            self.latest_ts = time.time()

    def get_frame_copy(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()