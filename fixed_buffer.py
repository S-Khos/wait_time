import numpy as np

class FixedBuffer:
    def __init__(self, max_len) -> None:
        self.max_len = max_len
        self.buffer = []

    def append(self, data) -> None:
        self.buffer.extend(data)
        self.buffer = self.buffer[-self.max_len:]

    def get_state(self) -> np.ndarray:
        return self.buffer

    def get_length(self) -> int:
        return len(self.buffer)