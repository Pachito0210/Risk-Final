import numpy as np

class CombatResolver:
    @staticmethod
    def roll(n_dice: int) -> list[int]:
        return sorted(np.random.randint(1, 7, n_dice), reverse=True)
    
    def resolve(self, attacker_troops: int, defender_troops: int):
        atk_dice = min(attacker_troops - 1, 3)
        def_dice = min(defender_troops, 2)
        atk = self.roll(atk_dice)
        dfn = self.roll(def_dice)
        
        atk_losses = def_losses = 0
        for a, d in zip(atk, dfn):
            if a > d:
                def_losses += 1
            else:
                atk_losses += 1
        return atk_losses, def_losses
    
    def win_probability(self, atk_troops: int, def_troops: int, simulations=5000):
        """Monte Carlo: P(atacante gana) dado N tropas vs M tropas"""
        wins = sum(
            self._simulate(atk_troops, def_troops)
            for _ in range(simulations)
        )
        return wins / simulations