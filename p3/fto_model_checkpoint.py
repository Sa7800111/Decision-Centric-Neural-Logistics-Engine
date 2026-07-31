import pickle
import os

class FTOCheckpoint:
    def __init__(self, directory):
        self.dir = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def save(self, model_state, epoch, metric_val):
        filename = f"fto_model_e{epoch}_m{metric_val:.4f}.pkl"
        path = os.path.join(self.dir, filename)
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        return path

    def load_best(self, metric_name="regret"):
        files = [f for f in os.listdir(self.dir) if f.endswith('.pkl')]
        if not files:
            return None
        best_file = sorted(files, key=lambda x: float(x.split('_m')[1].split('.pkl')[0]))[0]
        with open(os.path.join(self.dir, best_file), 'rb') as f:
            return pickle.load(f)