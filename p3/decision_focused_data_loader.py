import numpy as np

class DecisionDataLoader:
    def __init__(self, features, targets, contexts, batch_size=32):
        self.x = np.asarray(features)
        self.y = np.asarray(targets)
        self.c = contexts
        self.bs = batch_size
        self.indices = np.arange(len(self.x))

    def get_batches(self, shuffle=True):
        if shuffle:
            np.random.shuffle(self.indices)
            
        for i in range(0, len(self.indices), self.bs):
            idx = self.indices[i:i + self.bs]
            yield self.x[idx], self.y[idx], [self.c[j] for j in idx]

    def __len__(self):
        return int(np.ceil(len(self.x) / self.bs))