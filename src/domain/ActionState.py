# state.py
from dataclasses import dataclass, field
from threading import Lock
from typing import List

@dataclass
class ActionState:
    picking: bool = False          # 是否开始检测
    start_detect: bool = False
    detect_class = ['cup','banana','ping pong ball','sports ball','apple','mobile phone']# 你可以扩展：例如 "move" / "rotate" / "cmd"
    _lock: Lock = field(default_factory=Lock, repr=False)

    def set_picking(self, value: bool):
        with self._lock:
            self.picking = value

    def is_picking(self) -> bool:
        with self._lock:
            return self.picking

    def set_start_detect(self, value: bool):
        with self._lock:
            self.start_detect = value

    def get_start_detect(self) -> bool:
        with self._lock:
            return self.start_detect

    def set_detect_class(self, detect_class: list[str]):
        with self._lock:
            self.detect_class = detect_class

    def get_detect_class(self) -> list[str]:
        with self._lock:
            return self.detect_class


# 这个就是“全局单例”
action_state = ActionState()
