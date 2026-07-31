class DeterministicSCMMapper:
    def __init__(self, equations):
        if not isinstance(equations, dict):
            raise TypeError()
        for k, v in equations.items():
            if not callable(v):
                raise TypeError()
        self.equations = equations

    def evaluate_state(self, u_vector):
        if not isinstance(u_vector, dict):
            raise TypeError()
            
        state = {}
        for var in self.equations.keys():
            if var in u_vector:
                state[var] = u_vector[var]
                
        progress = True
        while progress:
            progress = False
            for var, func in self.equations.items():
                if var not in state:
                    try:
                        val = func(state, u_vector)
                        state[var] = val
                        progress = True
                    except KeyError:
                        continue
                        
        if len(state) != len(self.equations):
            raise RuntimeError()
            
        return state