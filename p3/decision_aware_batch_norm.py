import numpy as np

class DecisionBatchNorm:
    def __init__(self, num_features, momentum=0.1):
        self.mu = np.zeros(num_features)
        self.var = np.ones(num_features)
        self.m = momentum

    def forward(self, x, decision_weights=None):
        if decision_weights is None:
            decision_weights = np.ones(len(x)) / len(x)
        
        batch_mu = np.average(x, axis=0, weights=decision_weights)
        batch_var = np.average((x - batch_mu)**2, axis=0, weights=decision_weights)
        
        self.mu = (1 - self.m) * self.mu + self.m * batch_mu
        self.var = (1 - self.m) * self.var + self.m * batch_var
        
        return (x - batch_mu) / np.sqrt(batch_var + 1e-5)