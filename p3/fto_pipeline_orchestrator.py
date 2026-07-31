class FTOOrchestrator:
    def __init__(self, predictor, solver, feedback_engine):
        self.predictor = predictor
        self.solver = solver
        self.feedback = feedback_engine

    def process_batch(self, x_batch, y_true_batch, context_batch):
        history = []
        for x, yt, ctx in zip(x_batch, y_true_batch, context_batch):
            yp = self.predictor.predict(x)
            z = self.solver.solve(yp, ctx)
            
            loss = self.feedback.compute_loss(yt, yp, z)
            grad = self.feedback.compute_grad(yt, yp, z)
            
            self.predictor.update(grad)
            history.append(loss)
            
        return history

    def save_state(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.predictor.get_weights(), f)