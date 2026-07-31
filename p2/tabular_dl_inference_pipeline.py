import numpy as np

class TabularInferencePipeline:
    def __init__(self, preprocessor, fourier_map, regressor):
        self.prep = preprocessor
        self.fmap = fourier_map
        self.reg = regressor

    def run_inference(self, raw_data):
        clean_data = self.prep.transform(raw_data)
        phi = self.fmap.map(clean_data)
        return self.reg.predict(phi)

    def estimate_policy_value(self, data, action_vec):
        preds = self.run_inference(data)
        return float(np.mean(preds * action_vec))