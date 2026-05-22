import numpy as np

class MarkovChain:
    def __init__(self, states: list, transition_matrix: np.ndarray):
        self.states = states
        self.P = transition_matrix  # shape: (n, n), filas suman 1
        self.current = 0
    
    def step(self) -> str:
        self.current = np.random.choice(len(self.states), p=self.P[self.current])
        return self.states[self.current]
    
    def stationary_distribution(self) -> np.ndarray:
        # Distribución estacionaria: eigenvector de P^T con eigenvalue 1
        vals, vecs = np.linalg.eig(self.P.T)
        stat = np.real(vecs[:, np.argmax(np.real(vals))])
        return stat / stat.sum()
    
class MarkovAI(Player):
    def __init__(self, chain: MarkovChain, world: WorldGraph):
        self.chain = chain
        self.world = world
    
    def choose_action(self, engine) -> dict:
        state = self.chain.step()             # estado actual de Markov
        if state in ("S0", "S1"):
            return self._defensive_move(engine)
        else:
            return self._aggressive_move(engine)
    
    def _aggressive_move(self, engine):
        # Busca el territorio enemigo más débil adyacente
        targets = [
            t for t in self._my_borders(engine)
            if engine.world.G.nodes[t]["troops"] < 3
        ]
        return {"action": "attack", "target": targets[0] if targets else None}