from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
        self.total = 0

    def add(self, value):
        if len(self.data) == self.window_size:
            oldest_value = self.data.popleft()
            self.total -= oldest_value

        self.data.append(value)
        self.total += value

    def average(self):
        if not self.data:
            return None
        return self.total / len(self.data)

    def clear(self):
        self.data = deque(maxlen=self.window_size)
        self.total = 0
