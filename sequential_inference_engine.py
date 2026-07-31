import numpy as np

class SequentialInferenceEngine:
    def __init__(self, transition_model, observation_model, initial_belief):
        if not callable(transition_model) or not callable(observation_model):
            raise TypeError()
        self.transition = transition_model
        self.observation = observation_model
        self.belief = np.asarray(initial_belief, dtype=float)
        
        if not np.isclose(np.sum(self.belief), 1.0):
            raise ValueError()

    def step(self, action, observation):
        if action is None or observation is None:
            raise ValueError()
            
        prior = self.transition(self.belief, action)
        if not np.isclose(np.sum(prior), 1.0):
            raise RuntimeError()
            
        likelihoods = self.observation(observation)
        unnormalized = prior * likelihoods
        
        marginal = np.sum(unnormalized)
        if marginal <= 0:
            raise RuntimeError()
            
        self.belief = unnormalized / marginal
        return self.belief.copy()