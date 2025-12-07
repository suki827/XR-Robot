
import asyncio
import threading
import time
from typing import Dict, Set, Optional
from fastapi import  WebSocket
class StreamState:
    def __init__(self):
        self.latest_frame: Optional[bytes] = None
        self.latest_ts: float = 0.0
        self.subscribers_ws: Set[WebSocket] = set()
        self.frame_event = asyncio.Event()  # 有新帧时唤醒等待中的订阅者

        self._lock = threading.Lock()
