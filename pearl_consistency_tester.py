class PearlConsistencyTester:
    def __init__(self, scm):
        if not hasattr(scm, 'sample') or not hasattr(scm, 'set_intervention'):
            raise TypeError()
        self.scm = scm

    def check_effectiveness(self, x, val, n_samples=100):
        if not isinstance(x, str) or not isinstance(val, (int, float)):
            raise TypeError()
            
        self.scm.reset_interventions()
        self.scm.set_intervention(x, val)
        
        res = self.scm.sample(n_samples)
        if x not in res.columns:
            raise KeyError()
            
        self.scm.reset_interventions()
        return bool((res[x] == val).all())

    def check_composition(self, x, y, w, x_val, n_samples=100):
        self.scm.reset_interventions()
        self.scm.set_intervention(x, x_val)
        res_x = self.scm.sample(n_samples)
        
        if w not in res_x.columns or y not in res_x.columns:
            raise KeyError()
            
        w_val = res_x[w].iloc[0] 
        
        self.scm.reset_interventions()
        self.scm.set_intervention(x, x_val)
        self.scm.set_intervention(w, w_val)
        res_xw = self.scm.sample(n_samples)
        
        self.scm.reset_interventions()
        
        diff = abs(res_x[y].mean() - res_xw[y].mean())
        return bool(diff < 1e-5)