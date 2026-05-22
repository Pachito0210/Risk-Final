"""
engine.py
Motor principal del juego Risk.

Orquesta:
  - Turnos y fases (draft → attack → fortify)
  - Jugadores humanos e IA
  - Integración con WorldGraph y CombatResolver
  - Condición de victoria
  - Registro de eventos
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from graph.world_graph import WorldGraph
from simulation.combat import CombatResolver
from simulation.markov_ai import MarkovAI, STATES


# ─────────────────────────────────────────────────────────────
#  Estructuras de datos
# ─────────────────────────────────────────────────────────────

PHASES = ["draft", "attack", "fortify"]

@dataclass
class PlayerInfo:
    player_id:  str
    name:       str
    color:      str
    is_ai:      bool
    ai_agent:   Optional[MarkovAI] = None
    eliminated: bool = False


@dataclass
class GameEvent:
    turn:    int
    phase:   str
    player:  str
    action:  str
    details: str

    def __str__(self):
        return f"[T{self.turn:02} {self.phase:8}] {self.player}: {self.action} — {self.details}"


# ─────────────────────────────────────────────────────────────
#  GameEngine
# ─────────────────────────────────────────────────────────────

class GameEngine:
    """
    Motor central del juego Risk.
    Maneja el ciclo completo de turnos e integra todos los módulos.
    """

    # Bonus de tropas por continente controlado
    CONTINENT_BONUS = {
        "North America": 5,
        "South America": 2,
        "Africa":        3,
    }

    def __init__(
        self,
        world: WorldGraph,
        players: list[PlayerInfo],
        seed: Optional[int] = None,
    ):
        self.world    = world
        self.players  = players
        self.rng      = np.random.default_rng(seed)
        self.resolver = CombatResolver(self.rng)

        self.turn          = 1
        self.phase_idx     = 0
        self.current_idx   = 0
        self.events: list[GameEvent] = []
        self.winner: Optional[str]   = None
        self.running = True

    # ── Propiedades ───────────────────────────────────────────

    @property
    def phase(self) -> str:
        return PHASES[self.phase_idx]

    @property
    def current_player(self) -> PlayerInfo:
        return self.players[self.current_idx]

    @property
    def active_players(self) -> list[PlayerInfo]:
        return [p for p in self.players if not p.eliminated]

    # ── Setup inicial ─────────────────────────────────────────

    def setup(self) -> None:
        """
        Distribuye territorios aleatoriamente entre jugadores
        y coloca tropas iniciales (3 por territorio).
        """
        territories = self.world.all_territories()
        shuffled    = list(territories)
        self.rng.shuffle(shuffled)

        active = self.active_players
        for i, tid in enumerate(shuffled):
            player = active[i % len(active)]
            self.world.assign_territory(tid, player.player_id, troops=1)

        self._log("setup", "sistema", "distribucion",
                  f"{len(territories)} territorios entre {len(active)} jugadores")

    # ── Cálculo de refuerzos ──────────────────────────────────

    def calculate_reinforcements(self, player_id: str) -> int:
        """
        Tropas disponibles al inicio del turno:
          - max(3, territorios_propios // 3)
          - + bonus por continentes controlados
        """
        owned = len(self.world.territories_by_player(player_id))
        base  = max(3, owned // 3)
        bonus = sum(
            v for c, v in self.CONTINENT_BONUS.items()
            if self.world.controls_continent(player_id, c)
        )
        return base + bonus

    # ── Acciones del jugador humano ───────────────────────────

    def human_reinforce(self, territory_id: str, troops: int) -> dict:
        """Coloca tropas en un territorio. Fase: draft."""
        p = self.current_player
        self.world.add_troops(territory_id, troops, p.player_id)
        self._log("draft", p.player_id, "reinforce",
                  f"+{troops} tropas en {territory_id}")
        return {"ok": True, "territory": territory_id, "troops": troops}

    def human_attack(
        self,
        from_tid: str,
        to_tid:   str,
        attacking_troops: Optional[int] = None,
    ) -> dict:
        """
        Ataca un territorio vecino. Fase: attack.

        attacking_troops: cuántas tropas envía el jugador al combate
                          (1 .. from.troops - 1).
                          Si es None usa todas las disponibles (from.troops).
                          El número de dados es min(attacking_troops, 3).

        Devuelve un dict con:
          attacker_won   : bool
          summary        : str  resumen legible
          round_log      : list de { atk_dice, def_dice, atk_losses, def_losses }
          atk_start      : int  tropas atacantes al inicio del combate
          def_start      : int  tropas defensoras al inicio
          atk_remaining  : int  tropas atacantes que sobrevivieron
          def_remaining  : int  tropas defensoras que sobrevivieron
          troops_sent    : int  tropas que el jugador eligió enviar
          from_name      : str  nombre del territorio atacante
          to_name        : str  nombre del territorio defensor
        """
        p = self.current_player

        # ── Validaciones ──────────────────────────────────────
        if self.world.owner(from_tid) != p.player_id:
            raise ValueError(f"No controlas '{from_tid}'")
        if self.world.owner(to_tid) == p.player_id:
            raise ValueError("No puedes atacarte a ti mismo")
        if not self.world.are_neighbors(from_tid, to_tid):
            raise ValueError(f"'{from_tid}' y '{to_tid}' no son territorios vecinos")

        t_from = self.world.troops(from_tid)
        t_def  = self.world.troops(to_tid)

        if t_from < 2:
            raise ValueError("Necesitas al menos 2 tropas para atacar")

        # ── Calcular tropas que entran al combate ─────────────
        max_attacking = t_from - 1   # siempre queda 1 en origen

        if attacking_troops is None:
            # Usa todas las disponibles (batalla a muerte)
            effective_atk_pool = t_from
        else:
            # El jugador eligió N tropas atacantes (1..max_attacking)
            clamped = max(1, min(int(attacking_troops), max_attacking))
            # resolve_battle espera "tropas totales en territorio" e internamente
            # usa min(atk_troops - 1, 3) dados; por eso sumamos 1
            effective_atk_pool = clamped + 1

        # ── Resolver batalla con log de rondas ────────────────
        result = self.resolver.resolve_battle(
            effective_atk_pool, t_def, log_rounds=True
        )

        # ── Actualizar tropas en el grafo ─────────────────────
        troops_not_fighting = t_from - (effective_atk_pool - 1) - 1
        # tropas que no entraron al combate (quedan en origen sin riesgo)

        if result.attacker_won:
            # Tropas que se mueven al territorio conquistado
            troops_to_move = max(1, result.atk_remaining - 1)
            self.world.conquer(from_tid, to_tid, p.player_id, troops_to_move)
            # En origen quedan: las que no lucharon + las que lucharon y sobrevivieron - las que se mueven
            survivors_in_origin = (troops_not_fighting
                                   + (result.atk_remaining - 1)
                                   - (troops_to_move - 1))
            self.world.G.nodes[from_tid]["troops"] = max(1, survivors_in_origin)
        else:
            # Solo bajas entre las tropas que combatieron
            survivors = result.atk_remaining - 1   # -1 porque effective_atk_pool = chosen + 1
            self.world.G.nodes[from_tid]["troops"] = max(1,
                troops_not_fighting + survivors + 1
            )

        # ── Verificar eliminación y victoria ─────────────────
        if result.attacker_won:
            prev_owner = to_tid  # ya fue conquistado; buscamos quién era dueño antes
            # _check_elimination necesita el antiguo dueño; lo rastreamos vía eventos
            # Como world.conquer ya actualizó el owner, buscamos en el log
            for ev in reversed(self.events):
                if "→" in ev.details and to_tid in ev.details:
                    break
            # Método seguro: revisar todos los jugadores por si alguno quedó sin territorios
            for pl in self.players:
                if pl.player_id != p.player_id and not pl.eliminated:
                    if not self.world.territories_by_player(pl.player_id):
                        pl.eliminated = True
                        self._log("event", pl.player_id, "eliminated",
                                  f"perdió su último territorio")

        self._check_winner()

        # ── Serializar log de rondas ──────────────────────────
        round_log = [
            {
                "atk_dice":   r.atk_dice,
                "def_dice":   r.def_dice,
                "atk_losses": r.atk_losses,
                "def_losses": r.def_losses,
            }
            for r in result.round_log
        ]

        detail = result.summary()
        self._log("attack", p.player_id, "attack", detail)

        from_name = self.world.G.nodes[from_tid].get("name", from_tid)
        to_name   = self.world.G.nodes[to_tid].get("name", to_tid)

        return {
            "ok":            True,
            "attacker_won":  result.attacker_won,
            "summary":       detail,
            "round_log":     round_log,
            "atk_start":     effective_atk_pool,
            "def_start":     t_def,
            "atk_remaining": result.atk_remaining,
            "def_remaining": result.def_remaining,
            "troops_sent":   effective_atk_pool - 1,
            "from_name":     from_name,
            "to_name":       to_name,
        }

    def human_fortify(self, from_tid: str, to_tid: str, troops: int) -> dict:
        """Mueve tropas entre territorios propios. Fase: fortify."""
        p = self.current_player
        self.world.move_troops(from_tid, to_tid, troops, p.player_id)
        self._log("fortify", p.player_id, "fortify",
                  f"{troops} tropas de {from_tid} → {to_tid}")
        return {"ok": True}

    def end_phase(self) -> dict:
        """Avanza a la siguiente fase o al siguiente turno."""
        self.phase_idx += 1
        if self.phase_idx >= len(PHASES):
            self.phase_idx = 0
            self._next_player()
        return {"phase": self.phase, "player": self.current_player.player_id}

    # ── Turno de la IA ────────────────────────────────────────

    def run_ai_turn(self, verbose: bool = True) -> dict:
        """
        Ejecuta el turno completo de la IA actual.
        La IA maneja sus tres fases automáticamente.
        """
        p = self.current_player
        if not p.is_ai or not p.ai_agent:
            return {"ok": False, "reason": "No es un turno de IA"}

        troops = self.calculate_reinforcements(p.player_id)
        log    = p.ai_agent.take_turn(self.world, troops, verbose=verbose)

        self._log("full_turn", p.player_id, "ai_turn",
                  f"estado={log['state']}, conquistas={log['conquered']}")
        self._check_winner()
        self._next_player()

        return {"ok": True, "log": log}

    # ── Bucle principal ───────────────────────────────────────

    def run_full_game(self, max_turns: int = 50, verbose: bool = True) -> Optional[str]:
        """
        Corre una partida completa entre agentes IA.
        Devuelve el player_id del ganador (o None si se agota el tiempo).
        """
        self.setup()

        if verbose:
            print(f"\n{'═'*55}")
            print(f"  INICIO DE PARTIDA — {len(self.active_players)} jugadores")
            print(f"{'═'*55}")
            self._print_state()

        for t in range(1, max_turns + 1):
            self.turn = t
            if verbose:
                print(f"\n{'─'*55}")
                print(f"  TURNO {t}")
                print(f"{'─'*55}")

            for player in list(self.active_players):
                if self.winner:
                    break
                if player.eliminated:
                    continue

                self.current_idx = self.players.index(player)

                if player.is_ai and player.ai_agent:
                    self.run_ai_turn(verbose=verbose)

            if self.winner:
                break

            if verbose:
                self._print_state()

        if verbose:
            print(f"\n{'═'*55}")
            if self.winner:
                print(f"  GANADOR: {self.winner} en turno {self.turn}")
            else:
                print(f"  Partida terminada por límite de turnos.")
            print(f"{'═'*55}")

        return self.winner

    # ── Estado serializable ───────────────────────────────────

    def get_state(self) -> dict:
        """Devuelve el estado completo del juego como dict serializable."""
        territories = {}
        for tid, data in self.world.G.nodes(data=True):
            territories[tid] = {
                "name":      data["name"],
                "continent": data["continent"],
                "owner":     data["owner"],
                "troops":    data["troops"],
                "pos":       list(data["pos"]),
                "neighbors": self.world.neighbors(tid),
            }

        players_info = []
        for p in self.players:
            owned = self.world.territories_by_player(p.player_id)
            players_info.append({
                "id":           p.player_id,
                "name":         p.name,
                "color":        p.color,
                "is_ai":        p.is_ai,
                "eliminated":   p.eliminated,
                "territories":  owned,
                "troops_total": sum(self.world.troops(t) for t in owned),
                "reinforcements": self.calculate_reinforcements(p.player_id),
                "ai_state":     (p.ai_agent.state_name if p.ai_agent else None),
            })

        return {
            "turn":        self.turn,
            "phase":       self.phase,
            "current":     self.current_player.player_id,
            "winner":      self.winner,
            "running":     self.running,
            "territories": territories,
            "players":     players_info,
            "borders":     list(self.world.G.edges()),
        }

    # ── Internos ──────────────────────────────────────────────

    def _next_player(self) -> None:
        active = self.active_players
        if not active:
            return
        current_id = self.current_player.player_id
        active_ids = [p.player_id for p in active]
        try:
            idx = active_ids.index(current_id)
        except ValueError:
            idx = 0
        self.current_idx = self.players.index(
            active[(idx + 1) % len(active)]
        )
        if self.current_idx == 0 or (
            self.players[self.current_idx].player_id == active_ids[0]
            and idx != 0
        ):
            self.turn += 1

    def _check_elimination(self, loser_id: str, territory: str) -> None:
        """Elimina un jugador si perdió todos sus territorios."""
        if not self.world.territories_by_player(loser_id):
            for p in self.players:
                if p.player_id == loser_id:
                    p.eliminated = True
                    self._log("event", loser_id, "eliminated",
                              f"perdió último territorio {territory}")
                    break

    def _check_winner(self) -> None:
        """Declara ganador si un jugador controla todos los territorios."""
        for p in self.active_players:
            owned = self.world.territories_by_player(p.player_id)
            if len(owned) == self.world.G.number_of_nodes():
                self.winner  = p.player_id
                self.running = False
                self._log("event", p.player_id, "victory",
                          f"controla todos los {len(owned)} territorios")
                return

    def _log(self, phase: str, player: str, action: str, details: str) -> None:
        self.events.append(GameEvent(self.turn, phase, player, action, details))

    def _print_state(self) -> None:
        for p in self.active_players:
            owned  = self.world.territories_by_player(p.player_id)
            troops = sum(self.world.troops(t) for t in owned)
            conts  = self.world.controlled_continents(p.player_id)
            ai_st  = f" [{p.ai_agent.state_name}]" if p.ai_agent else ""
            print(f"  {p.name:12}{ai_st:14} | {len(owned):2} terr. | {troops:3} tropas | conts: {conts or '—'}")
