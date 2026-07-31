import numpy as np

class SemanticLogicGate:
    def __init__(self, activation_threshold=0.5):
        self.threshold = float(activation_threshold)

    def _binarize(self, value):
        val = np.asarray(value, dtype=float)
        return (val >= self.threshold).astype(int)

    def AND_causal(self, input_a, input_b):
        a = self._binarize(input_a)
        b = self._binarize(input_b)
        return np.minimum(a, b)

    def OR_causal(self, input_a, input_b):
        a = self._binarize(input_a)
        b = self._binarize(input_b)
        return np.maximum(a, b)

    def NOT_causal(self, input_val):
        val = self._binarize(input_val)
        return 1 - val

    def XOR_causal(self, input_a, input_b):
        a = self._binarize(input_a)
        b = self._binarize(input_b)
        return np.abs(a - b)