from collections import deque

class StageSmoother:
    def __init__(self, window=5):
        self.window = window
        self.buffer = deque(maxlen=window)

    def update(self, stage):
        self.buffer.append(stage)
        return max(set(self.buffer), key=self.buffer.count)
