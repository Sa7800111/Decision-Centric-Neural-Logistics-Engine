import numpy as np

class DecisionCVIterator:
    def __init__(self, n_splits=5):
        self.n = n_splits

    def split_data(self, x, y, contexts):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        fold_size = len(x) // self.n
        
        for i in range(self.n):
            test_idx = indices[i*fold_size : (i+1)*fold_size]
            train_idx = np.setdiff1d(indices, test_idx)
            yield train_idx, test_idx

    def evaluate_fold(self, train_idx, test_idx, model, solver):
        model.fit(train_idx)
        y_p = model.predict(test_idx)
        regrets = [solver.regret(y, yp) for y, yp in zip(y[test_idx], y_p)]
        return np.mean(regrets)