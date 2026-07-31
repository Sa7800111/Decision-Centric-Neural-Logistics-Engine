import numpy as np

class DecisionDataBuffer:
    def __init__(self, max_size=1000):
        self.size = max_size
        self.buffer = []

    def push(self, experience):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def clear(self):
        self.buffer = []