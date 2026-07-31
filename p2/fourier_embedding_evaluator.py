import numpy as np

class FourierEmbeddingEvaluator:
    def __init__(self, original_data, embedded_data):
        self.x = np.asarray(original_data)
        self.z = np.asarray(embedded_data)

    def compute_reconstruction_error(self):
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0).fit(self.z, self.x)
        preds = model.predict(self.z)
        return np.mean((self.x - preds)**2)

    def compute_feature_correlation(self):
        corr_matrix = np.corrcoef(self.z, rowvar=False)
        return np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))