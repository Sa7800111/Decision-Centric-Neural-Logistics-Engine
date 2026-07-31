import random
import numpy as np

class CasinoProbProgram:
    def __init__(self, house_edge=0.05):
        self.win_rate = 0.5 - house_edge

    def simulate_betting_sequence(self, initial_bankroll, n_rounds):
        bankroll = float(initial_bankroll)
        history = [bankroll]
        
        for _ in range(n_rounds):
            if bankroll <= 0:
                break
            bet = bankroll * 0.1
            if random.random() < self.win_rate:
                bankroll += bet
            else:
                bankroll -= bet
            history.append(bankroll)
            
        return np.array(history)