import numpy as np

class DecisionFeedbackLoop:
    def __init__(self, predictor, optimizer, loss_engine):
        self.predictor = predictor
        self.optimizer = optimizer
        self.loss_engine = loss_engine

    def run_iteration(self, x, y_true, context):
        y_pred = self.predictor.predict(x)
        decision = self.optimizer.solve(y_pred, context)
        
        regret = self.loss_engine.compute_regret(y_true, y_pred, context)
        grad = self.loss_engine.compute_decision_grad(y_true, y_pred, decision)
        
        self.predictor.update(x, grad)
        return float(regret)