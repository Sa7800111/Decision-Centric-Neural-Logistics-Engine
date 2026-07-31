import numpy as np

class JointMatrixEngine:
    def __init__(self, transition_matrices=None):
        self.matrices = []
        if transition_matrices is not None:
            for mat in transition_matrices:
                self.add_transition_matrix(mat)

    def add_transition_matrix(self, matrix):
        mat = np.asarray(matrix, dtype=float)
        if mat.ndim != 2:
            raise ValueError("Transition matrices must be 2D")
        
        if self.matrices:
            last_shape = self.matrices[-1].shape
            if last_shape[1] != mat.shape[0]:
                raise ValueError("Matrix dimensions do not align for multiplication")
                
        self.matrices.append(mat)

    def compute_joint_transition(self):
        if not self.matrices:
            raise RuntimeError("No matrices loaded")
            
        result = self.matrices[0]
        for mat in self.matrices[1:]:
            result = result @ mat
        return result

    def project_state(self, initial_state_vector):
        vec = np.asarray(initial_state_vector, dtype=float)
        if vec.ndim != 1:
            raise ValueError("Initial state must be a 1D vector")
            
        joint_transition = self.compute_joint_transition()
        
        if vec.shape[0] != joint_transition.shape[0]:
            raise ValueError("State vector dimension does not match transition matrix")
            
        return vec @ joint_transition