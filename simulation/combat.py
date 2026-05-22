"""
combat.py
Sistema de combate de Risk usando cadenas de Markov y numpy.

Modelo matemático
─────────────────
Un combate es una cadena de Markov donde cada estado es el par
(a, d): tropas del atacante y del defensor.

  S = { (a, d) | a >= 1, d >= 0 }

Estados absorbentes:
  - (1, d) con d > 0  → defensor gana (atacante no puede atacar con 1 tropa)
  - (a, 0) con a > 1  → atacante conquista

En cada ronda se tiran dados:
  - Atacante: min(a-1, 3) dados
  - Defensor: min(d,   2) dados

Se comparan el mayor dado del atacante vs el mayor del defensor,
y si hay 2 pares, también el segundo mayor vs segundo mayor.
El defensor gana los empates.

Probabilidades exactas (3 atk vs 2 def):
  P(atk pierde 2) ≈ 0.4483
  P(empate 1-1)   ≈ 0.2276
  P(def pierde 2) ≈ 0.3241
"""

import numpy as np
from itertools import product
from dataclasses import dataclass, field
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────
#  Estructuras de resultado
# ─────────────────────────────────────────────────────────────

@dataclass
class RoundResult:
    """Resultado de una sola ronda de dados."""
    atk_dice:  list[int]
    def_dice:  list[int]
    atk_losses: int
    def_losses: int

    def __str__(self):
        return (
            f"ATK {self.atk_dice} vs DEF {self.def_dice} → "
            f"ATK pierde {self.atk_losses} | DEF pierde {self.def_losses}"
        )


@dataclass
class BattleResult:
    """Resultado de una batalla completa (múltiples rondas)."""
    attacker_won:   bool
    rounds_played:  int
    atk_start:      int
    def_start:      int
    atk_remaining:  int
    def_remaining:  int
    atk_total_losses: int
    def_total_losses: int
    round_log:      list[RoundResult] = field(default_factory=list)

    @property
    def atk_loss_rate(self) -> float:
        return self.atk_total_losses / max(self.atk_start - 1, 1)

    @property
    def def_loss_rate(self) -> float:
        return self.def_total_losses / max(self.def_start, 1)

    def summary(self) -> str:
        winner = "ATACANTE" if self.attacker_won else "DEFENSOR"
        return (
            f"Ganador: {winner} | "
            f"Rondas: {self.rounds_played} | "
            f"ATK: {self.atk_start}→{self.atk_remaining} | "
            f"DEF: {self.def_start}→{self.def_remaining}"
        )


@dataclass
class BattleStats:
    """Estadísticas agregadas de múltiples batallas simuladas."""
    atk_troops:       int
    def_troops:       int
    simulations:      int
    atk_wins:         int
    def_wins:         int
    avg_rounds:       float
    avg_atk_losses:   float
    avg_def_losses:   float
    win_probability:  float          # P(atacante gana)
    markov_win_prob:  Optional[float] = None   # calculado analíticamente

    def __str__(self):
        return (
            f"\n{'═'*52}\n"
            f"  Estadísticas de batalla: {self.atk_troops} ATK vs {self.def_troops} DEF\n"
            f"{'─'*52}\n"
            f"  Simulaciones          : {self.simulations:,}\n"
            f"  Victoria atacante     : {self.atk_wins:,}  ({self.win_probability*100:.1f}%)\n"
            f"  Victoria defensor     : {self.def_wins:,}  ({(1-self.win_probability)*100:.1f}%)\n"
            f"  Prob. analítica Markov: {self.markov_win_prob*100:.1f}%\n"
            f"  Rondas promedio       : {self.avg_rounds:.2f}\n"
            f"  Pérdidas ATK prom.    : {self.avg_atk_losses:.2f}\n"
            f"  Pérdidas DEF prom.    : {self.avg_def_losses:.2f}\n"
            f"{'═'*52}"
        )


# ─────────────────────────────────────────────────────────────
#  Probabilidades exactas (combinatoria)
# ─────────────────────────────────────────────────────────────

def _compute_round_probs(n_atk: int, n_def: int) -> dict[tuple, float]:
    """
    Calcula P(atk_losses, def_losses) para una ronda dados n_atk y n_def dados.
    Enumera todos los resultados posibles de los dados (6^n combinaciones).
    Devuelve un dict {(atk_losses, def_losses): probabilidad}.
    """
    outcomes: dict[tuple, int] = {}
    total = 0

    for atk_rolls in product(range(1, 7), repeat=n_atk):
        for def_rolls in product(range(1, 7), repeat=n_def):
            atk_sorted = sorted(atk_rolls, reverse=True)
            def_sorted = sorted(def_rolls, reverse=True)

            atk_loss = def_loss = 0
            for a, d in zip(atk_sorted, def_sorted):
                if a > d:
                    def_loss += 1
                else:
                    atk_loss += 1  # defensor gana empates

            key = (atk_loss, def_loss)
            outcomes[key] = outcomes.get(key, 0) + 1
            total += 1

    return {k: v / total for k, v in outcomes.items()}


# Caché de probabilidades por (n_atk_dice, n_def_dice)
_PROB_CACHE: dict[tuple, dict] = {}

def get_round_probs(atk_troops: int, def_troops: int) -> dict[tuple, float]:
    """Devuelve probabilidades de ronda usando caché."""
    n_atk = min(atk_troops - 1, 3)
    n_def = min(def_troops, 2)
    key = (n_atk, n_def)
    if key not in _PROB_CACHE:
        _PROB_CACHE[key] = _compute_round_probs(n_atk, n_def)
    return _PROB_CACHE[key]


# ─────────────────────────────────────────────────────────────
#  CombatResolver — simulación ronda a ronda
# ─────────────────────────────────────────────────────────────

class CombatResolver:
    """
    Resuelve combates de Risk simulando ronda a ronda con numpy.
    Integra con WorldGraph para actualizar el estado del mapa.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    # ── Dados ─────────────────────────────────────────────────

    def roll_dice(self, n: int) -> list[int]:
        """Lanza n dados y devuelve los resultados en orden descendente."""
        return sorted(self.rng.integers(1, 7, size=n).tolist(), reverse=True)

    # ── Una ronda ─────────────────────────────────────────────

    def resolve_round(self, atk_troops: int, def_troops: int) -> RoundResult:
        """
        Simula una única ronda de combate.
        atk_troops: tropas del atacante en su territorio (debe ser >= 2)
        def_troops: tropas del defensor (debe ser >= 1)
        """
        n_atk = min(atk_troops - 1, 3)
        n_def = min(def_troops, 2)

        atk_dice = self.roll_dice(n_atk)
        def_dice = self.roll_dice(n_def)

        atk_loss = def_loss = 0
        for a, d in zip(atk_dice, def_dice):
            if a > d:
                def_loss += 1
            else:
                atk_loss += 1

        return RoundResult(atk_dice, def_dice, atk_loss, def_loss)

    # ── Batalla completa ──────────────────────────────────────

    def resolve_battle(
        self,
        atk_troops: int,
        def_troops: int,
        log_rounds: bool = False,
    ) -> BattleResult:
        """
        Simula una batalla completa hasta que alguien gane.
        Devuelve BattleResult con todas las estadísticas.

        La batalla termina cuando:
          - atk_troops == 1: atacante no puede continuar → defensor gana
          - def_troops == 0: territorio conquistado → atacante gana
        """
        if atk_troops < 2:
            raise ValueError("El atacante necesita al menos 2 tropas para atacar.")
        if def_troops < 1:
            raise ValueError("El defensor necesita al menos 1 tropa.")

        a = atk_troops
        d = def_troops
        rounds = 0
        log = []

        while a > 1 and d > 0:
            result = self.resolve_round(a, d)
            a -= result.atk_losses
            d -= result.def_losses
            rounds += 1
            if log_rounds:
                log.append(result)

        return BattleResult(
            attacker_won    = (d == 0),
            rounds_played   = rounds,
            atk_start       = atk_troops,
            def_start       = def_troops,
            atk_remaining   = a,
            def_remaining   = d,
            atk_total_losses = atk_troops - a,
            def_total_losses = def_troops  - d,
            round_log       = log,
        )

    # ── Integración con WorldGraph ────────────────────────────

    def attack_territory(
        self,
        world,
        from_tid: str,
        to_tid:   str,
        attacker_id: str,
        troops_to_move: int = 1,
        log_rounds: bool = False,
    ) -> BattleResult:
        """
        Ejecuta un ataque entre dos territorios del WorldGraph.
        Si el atacante gana, actualiza el grafo (conquer).
        Si pierde, solo descuenta las bajas.

        Args:
            world          : instancia de WorldGraph
            from_tid       : territorio atacante
            to_tid         : territorio defensor
            attacker_id    : player_id del atacante
            troops_to_move : tropas que ocupan el territorio si se conquista
            log_rounds     : registrar cada ronda en BattleResult
        """
        from graph.world_graph import NotNeighborError, NotOwnerError

        # Validaciones de grafo
        if not world.are_neighbors(from_tid, to_tid):
            raise NotNeighborError(f"'{from_tid}' y '{to_tid}' no son vecinos.")
        if world.owner(from_tid) != attacker_id:
            raise NotOwnerError(f"'{attacker_id}' no controla '{from_tid}'.")
        if world.owner(to_tid) == attacker_id:
            raise ValueError("No puedes atacar tu propio territorio.")

        atk_troops = world.troops(from_tid)
        def_troops = world.troops(to_tid)

        result = self.resolve_battle(atk_troops, def_troops, log_rounds)

        if result.attacker_won:
            # Asegurar que troops_to_move sea válido
            troops_to_move = max(1, min(troops_to_move, result.atk_remaining - 1))
            world.conquer(from_tid, to_tid, attacker_id, troops_to_move)
            # Ajustar tropas restantes del atacante
            world.G.nodes[from_tid]["troops"] = result.atk_remaining - troops_to_move
        else:
            # Solo bajas del atacante
            world.G.nodes[from_tid]["troops"] = result.atk_remaining

        return result


# ─────────────────────────────────────────────────────────────
#  MarkovCombat — análisis analítico con matriz de transición
# ─────────────────────────────────────────────────────────────

class MarkovCombat:
    """
    Modela una batalla de Risk como cadena de Markov.

    Los estados son pares (a, d) con a >= 1, d >= 0.
    La matriz de transición P[s][s'] = probabilidad de ir de estado s a s'.
    Estados absorbentes: (1, d>0) → defensor gana; (a, 0) → atacante gana.

    Permite calcular:
      - P(atacante gana) de forma exacta
      - Distribución de tropas restantes
      - Número esperado de rondas
    """

    def __init__(self, max_troops: int = 20):
        self.max_troops = max_troops
        self._cache: dict[tuple, float] = {}   # (a, d) → P(atk gana)

    def _enumerate_states(self, atk_start: int, def_start: int):
        """
        Enumera todos los estados alcanzables desde (atk_start, def_start).
        Devuelve lista de estados y sus índices.
        """
        states = []
        for a in range(1, atk_start + 1):
            for d in range(0, def_start + 1):
                states.append((a, d))
        return states

    def win_probability(self, atk_troops: int, def_troops: int) -> float:
        """
        Calcula P(atacante gana) de forma exacta usando la cadena de Markov.
        Usa programación dinámica (memoización) sobre los estados.

        P(a, d) = sum over outcomes: p(outcome) * P(a', d')
        con condiciones base:
          P(a, 0) = 1.0  para todo a > 1
          P(1, d) = 0.0  para todo d > 0
        """
        key = (atk_troops, def_troops)
        if key in self._cache:
            return self._cache[key]

        # Condiciones base
        if def_troops == 0:
            return 1.0
        if atk_troops == 1:
            return 0.0

        # Obtener probabilidades de ronda
        probs = get_round_probs(atk_troops, def_troops)

        total = 0.0
        for (atk_loss, def_loss), p in probs.items():
            a_next = atk_troops - atk_loss
            d_next = def_troops - def_loss
            total += p * self.win_probability(a_next, d_next)

        self._cache[key] = total
        return total

    def expected_losses(self, atk_troops: int, def_troops: int) -> dict:
        """
        Calcula las pérdidas esperadas de cada bando mediante simulación
        de la distribución completa de estados finales.

        Devuelve:
          {
            'atk_expected_loss': float,
            'def_expected_loss': float,
            'win_prob': float,
            'state_distribution': { (a_final, d_final): probabilidad }
          }
        """
        # Propagación de distribución de probabilidad
        # dist[(a,d)] = P(estar en estado (a,d))
        dist: dict[tuple, float] = {(atk_troops, def_troops): 1.0}
        final_dist: dict[tuple, float] = {}

        while dist:
            next_dist: dict[tuple, float] = {}
            for (a, d), p_state in dist.items():
                # Estado absorbente
                if d == 0 or a == 1:
                    final_dist[(a, d)] = final_dist.get((a, d), 0) + p_state
                    continue
                # Transición
                round_probs = get_round_probs(a, d)
                for (atk_loss, def_loss), p_trans in round_probs.items():
                    a2 = a - atk_loss
                    d2 = d - def_loss
                    next_dist[(a2, d2)] = next_dist.get((a2, d2), 0) + p_state * p_trans
            dist = next_dist

        # Calcular estadísticas de la distribución final
        atk_exp_loss = sum(
            (atk_troops - a) * p
            for (a, d), p in final_dist.items()
        )
        def_exp_loss = sum(
            (def_troops - d) * p
            for (a, d), p in final_dist.items()
        )
        win_prob = sum(
            p for (a, d), p in final_dist.items() if d == 0
        )

        return {
            "atk_expected_loss": atk_exp_loss,
            "def_expected_loss": def_exp_loss,
            "win_prob":          win_prob,
            "state_distribution": final_dist,
        }

    def transition_matrix(self, atk_troops: int, def_troops: int) -> tuple:
        """
        Construye la matriz de transición numpy P para el combate (a, d).
        Útil para análisis algebraico y para la entrega universitaria.

        Devuelve:
          (P, states, state_index)
          P       : np.ndarray de forma (n_states, n_states)
          states  : lista de estados (a, d)
          idx     : dict {(a,d): índice en la matriz}
        """
        states = self._enumerate_states(atk_troops, def_troops)
        idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        P = np.zeros((n, n))

        for (a, d) in states:
            i = idx[(a, d)]
            # Absorbentes: se quedan en sí mismos (P[i][i] = 1)
            if d == 0 or a == 1:
                P[i][i] = 1.0
                continue
            round_probs = get_round_probs(a, d)
            for (atk_loss, def_loss), p in round_probs.items():
                j = idx[(a - atk_loss, d - def_loss)]
                P[i][j] += p

        return P, states, idx


# ─────────────────────────────────────────────────────────────
#  CombatSimulator — estadísticas de múltiples batallas
# ─────────────────────────────────────────────────────────────

class CombatSimulator:
    """
    Corre N batallas Monte Carlo y devuelve estadísticas agregadas.
    Complementa el análisis analítico de MarkovCombat.
    """

    def __init__(self, seed: Optional[int] = None):
        self.resolver = CombatResolver(np.random.default_rng(seed))
        self.markov   = MarkovCombat()

    def run(
        self,
        atk_troops: int,
        def_troops: int,
        n: int = 5000,
    ) -> BattleStats:
        """
        Simula n batallas y devuelve BattleStats con estadísticas completas.
        Incluye la probabilidad analítica de Markov como referencia.
        """
        atk_wins     = 0
        total_rounds = 0
        total_atk_loss = 0
        total_def_loss = 0

        for _ in range(n):
            r = self.resolver.resolve_battle(atk_troops, def_troops)
            if r.attacker_won:
                atk_wins += 1
            total_rounds   += r.rounds_played
            total_atk_loss += r.atk_total_losses
            total_def_loss += r.def_total_losses

        markov_prob = self.markov.win_probability(atk_troops, def_troops)

        return BattleStats(
            atk_troops      = atk_troops,
            def_troops      = def_troops,
            simulations     = n,
            atk_wins        = atk_wins,
            def_wins        = n - atk_wins,
            avg_rounds      = total_rounds   / n,
            avg_atk_losses  = total_atk_loss / n,
            avg_def_losses  = total_def_loss / n,
            win_probability = atk_wins / n,
            markov_win_prob = markov_prob,
        )

    def probability_table(self, max_atk: int = 8, max_def: int = 6) -> np.ndarray:
        """
        Genera tabla numpy de P(atk gana) para combinaciones de tropas.
        Filas = tropas atacante (2..max_atk)
        Cols  = tropas defensor (1..max_def)
        """
        rows = range(2, max_atk + 1)
        cols = range(1, max_def + 1)
        table = np.zeros((len(rows), len(cols)))

        for i, a in enumerate(rows):
            for j, d in enumerate(cols):
                table[i][j] = self.markov.win_probability(a, d)

        return table
