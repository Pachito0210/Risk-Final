"""
markov_ai.py
IA enemiga basada en cadenas de Markov.

Modelo matemático
─────────────────
La IA modela su situación como una cadena de Markov con 4 estados:

  S0 = EQUILIBRIO   — ningún jugador domina claramente
  S1 = EXPANSION    — la IA controla al menos un continente
  S2 = DOMINANCIA   — la IA tiene >50% de territorios
  S3 = CRITICO      — la IA tiene <30% de territorios (defensiva)

Matriz de transición P (4x4):
Cada fila es el estado actual, cada columna el siguiente estado.
Las probabilidades se actualizan en cada turno según el estado real del juego.

Estrategia por estado:
  S0 → Consolidar fronteras, atacar territorios débiles
  S1 → Expandir agresivamente hacia nuevos continentes
  S2 → Atacar al jugador más fuerte para evitar que gane
  S3 → Defender territorios propios, reagrupar tropas
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import Optional
from simulation.combat import MarkovCombat


# ─────────────────────────────────────────────────────────────
#  Estados de la IA
# ─────────────────────────────────────────────────────────────

STATES = ["EQUILIBRIO", "EXPANSION", "DOMINANCIA", "CRITICO"]
S_EQUIL = 0
S_EXPAN = 1
S_DOMIN = 2
S_CRITI = 3

# Matriz de transición base P[estado_actual][estado_siguiente]
# Refleja la dinámica natural del juego:
#   - Desde EQUILIBRIO es probable ir a EXPANSION o CRITICO
#   - Desde EXPANSION es probable consolidar (DOMINANCIA) o retroceder
#   - Desde DOMINANCIA es difícil caer a CRITICO directamente
#   - Desde CRITICO se puede recuperar lentamente
BASE_TRANSITION = np.array([
    # EQ     EX     DO     CR
    [0.30,  0.40,  0.10,  0.20],   # S0: EQUILIBRIO
    [0.20,  0.35,  0.35,  0.10],   # S1: EXPANSION
    [0.10,  0.25,  0.55,  0.10],   # S2: DOMINANCIA
    [0.30,  0.10,  0.05,  0.55],   # S3: CRITICO
], dtype=float)


@dataclass
class AIAction:
    """Acción que decide tomar la IA en su turno."""
    action_type: str          # "reinforce" | "attack" | "fortify" | "pass"
    source: Optional[str]     # territorio origen
    target: Optional[str]     # territorio destino
    troops: int               # tropas a usar
    priority: float           # score de la decisión (mayor = mejor)
    reason: str               # explicación legible

    def __str__(self):
        return (
            f"[{self.action_type.upper():10}] "
            f"{self.source or '—':4} → {self.target or '—':4} | "
            f"{self.troops} tropas | {self.reason}"
        )


# ─────────────────────────────────────────────────────────────
#  MarkovAI
# ─────────────────────────────────────────────────────────────

class MarkovAI:
    """
    Agente IA que usa una cadena de Markov para modelar
    su estado estratégico y tomar decisiones de juego.

    Integra MarkovCombat para evaluar probabilidades antes de atacar.
    """

    def __init__(
        self,
        player_id: str,
        transition_matrix: Optional[np.ndarray] = None,
        min_win_prob: float = 0.45,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Args:
            player_id        : id del jugador IA
            transition_matrix: matriz P personalizada (usa BASE si None)
            min_win_prob     : probabilidad mínima para decidir atacar
            rng              : generador numpy para reproducibilidad
        """
        self.player_id    = player_id
        self.P            = transition_matrix if transition_matrix is not None \
                            else BASE_TRANSITION.copy()
        self.min_win_prob = min_win_prob
        self.rng          = rng or np.random.default_rng()
        self.markov_combat = MarkovCombat()

        self.current_state = S_EQUIL
        self.state_history: list[int] = [S_EQUIL]
        self.turn_count    = 0

    # ── Estado de Markov ──────────────────────────────────────

    def update_state(self, world) -> int:
        """
        Evalúa el tablero y actualiza el estado de Markov de la IA.
        El nuevo estado depende de la situación real del juego Y
        de la transición estocástica de la cadena.

        Devuelve el nuevo estado.
        """
        total   = world.G.number_of_nodes()
        owned   = len(world.territories_by_player(self.player_id))
        ratio   = owned / total if total > 0 else 0
        conts   = world.controlled_continents(self.player_id)

        # Determinar estado "real" según el tablero
        if ratio == 0:
            real_state = S_CRITI
        elif ratio < 0.30:
            real_state = S_CRITI
        elif conts:
            real_state = S_EXPAN if ratio < 0.50 else S_DOMIN
        else:
            real_state = S_EQUIL

        # Transición estocástica: combina estado real con cadena de Markov
        # 70% peso al estado real, 30% a la cadena (añade variabilidad)
        probs_markov = self.P[self.current_state]
        probs_real   = np.zeros(4)
        probs_real[real_state] = 1.0
        combined = 0.70 * probs_real + 0.30 * probs_markov

        self.current_state = int(self.rng.choice(4, p=combined))
        self.state_history.append(self.current_state)
        self.turn_count += 1
        return self.current_state

    @property
    def state_name(self) -> str:
        return STATES[self.current_state]

    def stationary_distribution(self) -> np.ndarray:
        """
        Distribución estacionaria π de la cadena de Markov.
        π = eigenvector de P^T con eigenvalue 1, normalizado.
        Indica la proporción de tiempo esperada en cada estado a largo plazo.
        """
        vals, vecs = np.linalg.eig(self.P.T)
        idx  = np.argmax(np.real(vals))
        stat = np.real(vecs[:, idx])
        return stat / stat.sum()

    # ── Decisiones por fase ───────────────────────────────────

    def decide_reinforce(self, world, available_troops: int) -> list[AIAction]:
        """
        Fase de refuerzo: decide dónde colocar tropas disponibles.
        Estrategia varía según el estado de Markov actual.

        Devuelve lista de AIActions ordenadas por prioridad.
        """
        owned = world.territories_by_player(self.player_id)
        if not owned:
            return []

        actions = []

        for tid in owned:
            score = 0.0
            reason_parts = []

            # Factor 1: territorios en frontera valen más
            enemies = world.enemy_neighbors(tid, self.player_id)
            if enemies:
                score += len(enemies) * 2.0
                reason_parts.append(f"{len(enemies)} enemigos adyacentes")

            # Factor 2: ajuste por estado Markov
            if self.current_state == S_CRITI:
                # Defensivo: reforzar el territorio con más amenaza
                threat = sum(world.troops(e) for e in enemies)
                score += threat * 0.5
                reason_parts.append("modo defensivo")

            elif self.current_state == S_EXPAN:
                # Expansivo: reforzar el que tiene más tropas propias (rampa de ataque)
                score += world.troops(tid) * 0.3
                reason_parts.append("modo expansión")

            elif self.current_state == S_DOMIN:
                # Dominante: distribuir hacia fronteras con continentes no conquistados
                cont = world.continent(tid)
                if not world.controls_continent(self.player_id, cont):
                    score += 3.0
                    reason_parts.append("completar continente")

            else:  # EQUILIBRIO
                score += len(enemies) * 1.5
                reason_parts.append("equilibrio")

            actions.append(AIAction(
                action_type="reinforce",
                source=None,
                target=tid,
                troops=0,          # se calculará al distribuir
                priority=score,
                reason=", ".join(reason_parts) if reason_parts else "base",
            ))

        # Ordenar por prioridad y distribuir tropas proporcionalmente
        actions.sort(key=lambda a: a.priority, reverse=True)
        total_score = sum(a.priority for a in actions) or 1.0
        remaining   = available_troops

        for i, action in enumerate(actions):
            if i == len(actions) - 1:
                action.troops = remaining
            else:
                share = int(available_troops * action.priority / total_score)
                share = max(0, min(share, remaining))
                action.troops = share
                remaining    -= share

        return [a for a in actions if a.troops > 0]

    def decide_attacks(self, world) -> list[AIAction]:
        """
        Fase de ataque: evalúa todos los posibles ataques y selecciona
        los más rentables según la probabilidad de victoria (MarkovCombat).

        Solo ataca si P(ganar) >= min_win_prob (ajustado por estado).
        """
        # Umbral dinámico según estado
        threshold_map = {
            S_EQUIL: self.min_win_prob,
            S_EXPAN: self.min_win_prob - 0.10,   # más agresivo
            S_DOMIN: self.min_win_prob - 0.05,
            S_CRITI: self.min_win_prob + 0.15,   # muy conservador
        }
        threshold = threshold_map.get(self.current_state, self.min_win_prob)

        actions = []
        owned = world.territories_by_player(self.player_id)

        for src in owned:
            atk_troops = world.troops(src)
            if atk_troops < 2:
                continue

            for dst in world.enemy_neighbors(src, self.player_id):
                def_troops = world.troops(dst)
                win_prob   = self.markov_combat.win_probability(atk_troops, def_troops)

                if win_prob < threshold:
                    continue

                # Score del ataque
                score = win_prob * 10

                # Bonus por conquistar continente
                cont = world.continent(dst)
                if not world.controls_continent(self.player_id, cont):
                    owned_in_cont = sum(
                        1 for t in world.territories_by_player(self.player_id)
                        if world.continent(t) == cont
                    )
                    total_in_cont = sum(
                        1 for t in world.all_territories()
                        if world.continent(t) == cont
                    )
                    # Si casi completa el continente, sube prioridad
                    score += (owned_in_cont / total_in_cont) * 5

                # Bonus por atacar al líder (estado DOMINANCIA)
                if self.current_state == S_DOMIN:
                    score += def_troops * 0.2

                losses = self.markov_combat.expected_losses(atk_troops, def_troops)
                reason = (
                    f"P(ganar)={win_prob:.0%}, "
                    f"pérd. esp. ATK={losses['atk_expected_loss']:.1f}"
                )

                actions.append(AIAction(
                    action_type="attack",
                    source=src,
                    target=dst,
                    troops=atk_troops - 1,
                    priority=score,
                    reason=reason,
                ))

        actions.sort(key=lambda a: a.priority, reverse=True)
        return actions

    def decide_fortify(self, world) -> Optional[AIAction]:
        """
        Fase de fortificación: mueve tropas desde territorio seguro
        hacia la frontera más expuesta.
        """
        owned    = world.territories_by_player(self.player_id)
        frontier = world.frontier_territories(self.player_id)

        if not frontier:
            return None

        # Territorio más expuesto (más enemigos adyacentes)
        best_dst = max(
            frontier,
            key=lambda t: sum(world.troops(e) for e in world.enemy_neighbors(t, self.player_id))
        )

        # Territorio seguro con más tropas que pueda ceder
        safe = [
            t for t in owned
            if not world.enemy_neighbors(t, self.player_id)
            and world.troops(t) > 1
            and world.are_neighbors(t, best_dst)
        ]

        if not safe:
            return None

        best_src = max(safe, key=lambda t: world.troops(t))
        movable  = world.troops(best_src) - 1

        if movable <= 0:
            return None

        return AIAction(
            action_type="fortify",
            source=best_src,
            target=best_dst,
            troops=movable,
            priority=1.0,
            reason=f"reforzar frontera expuesta",
        )

    # ── Turno completo ────────────────────────────────────────

    def take_turn(self, world, available_troops: int, verbose: bool = True) -> dict:
        """
        Ejecuta el turno completo de la IA:
          1. Actualiza estado de Markov
          2. Decide y aplica refuerzos
          3. Decide y aplica ataques
          4. Decide y aplica fortificación

        Devuelve resumen del turno.
        """
        from simulation.combat import CombatResolver
        resolver = CombatResolver(self.rng)

        state = self.update_state(world)
        log   = {
            "player":    self.player_id,
            "state":     STATES[state],
            "reinforce": [],
            "attacks":   [],
            "fortify":   None,
            "conquered": [],
        }

        if verbose:
            print(f"\n  [{self.player_id}] Estado Markov: {STATES[state]}")

        # ── Refuerzo ──────────────────────────────────────────
        reinforce_actions = self.decide_reinforce(world, available_troops)
        for action in reinforce_actions:
            world.add_troops(action.target, action.troops, self.player_id)
            log["reinforce"].append(action)
            if verbose:
                print(f"  ↑ {action}")

        # ── Ataques ───────────────────────────────────────────
        attack_actions = self.decide_attacks(world)
        for action in attack_actions[:3]:   # máximo 3 ataques por turno
            if world.troops(action.source) < 2:
                continue
            # Revalidar: el destino pudo conquistarse en este mismo turno
            if world.owner(action.target) == self.player_id:
                continue
            result = resolver.attack_territory(
                world,
                from_tid=action.source,
                to_tid=action.target,
                attacker_id=self.player_id,
                troops_to_move=max(1, world.troops(action.source) // 2),
            )
            log["attacks"].append((action, result))
            if result.attacker_won:
                log["conquered"].append(action.target)
            if verbose:
                print(f"  ⚔  {action}")
                print(f"     → {result.summary()}")

        # ── Fortificación ─────────────────────────────────────
        fortify = self.decide_fortify(world)
        if fortify:
            try:
                world.move_troops(
                    fortify.source, fortify.target,
                    fortify.troops, self.player_id
                )
                log["fortify"] = fortify
                if verbose:
                    print(f"  → {fortify}")
            except Exception:
                pass

        return log

    # ── Análisis ──────────────────────────────────────────────

    def state_report(self) -> str:
        """Resumen del historial de estados Markov de la IA."""
        hist = self.state_history
        counts = {s: hist.count(i) for i, s in enumerate(STATES)}
        stat   = self.stationary_distribution()

        lines = [
            f"  Jugador: {self.player_id}",
            f"  Turnos jugados: {self.turn_count}",
            f"  Estado actual: {STATES[self.current_state]}",
            f"",
            f"  Historial de estados:",
        ]
        for i, s in enumerate(STATES):
            pct  = counts[s] / len(hist) * 100 if hist else 0
            bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"    {s:12} {bar} {pct:5.1f}%  (estacionario: {stat[i]:.1%})")

        return "\n".join(lines)
