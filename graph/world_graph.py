"""
world_graph.py
Sistema de grafos del juego Risk usando networkx.

Cada territorio es un nodo con atributos:
  - name       : nombre largo del territorio
  - continent  : continente al que pertenece
  - owner      : id del jugador dueño (None = sin dueño)
  - troops     : cantidad de tropas presentes
  - pos        : (x, y) para visualización

Cada frontera es una arista no dirigida.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from typing import Optional
from data.map_data import CONTINENTS, BORDERS


# ─────────────────────────────────────────────────────────────
#  Excepciones del dominio
# ─────────────────────────────────────────────────────────────

class TerritoryError(Exception):
    """Error relacionado con operaciones sobre territorios."""


class NotNeighborError(TerritoryError):
    """Dos territorios no son vecinos."""


class InsufficientTroopsError(TerritoryError):
    """No hay tropas suficientes para la operación."""


class NotOwnerError(TerritoryError):
    """El jugador no controla el territorio."""


# ─────────────────────────────────────────────────────────────
#  WorldGraph
# ─────────────────────────────────────────────────────────────

class WorldGraph:
    """
    Grafo del mundo Risk.

    Usa nx.Graph internamente:
      - Nodos  → territorios (con atributos como metadatos)
      - Aristas → fronteras navegables entre territorios
    """

    def __init__(self):
        self.G: nx.Graph = nx.Graph()
        self._build()

    # ── Construcción ──────────────────────────────────────────

    def _build(self) -> None:
        """Construye el grafo a partir de map_data."""
        for continent, data in CONTINENTS.items():
            for tid, name, x, y in data["territories"]:
                self.G.add_node(
                    tid,
                    name=name,
                    continent=continent,
                    owner=None,
                    troops=0,
                    pos=(x, y),
                )

        for a, b in BORDERS:
            self.G.add_edge(a, b)

    # ── Validaciones internas ─────────────────────────────────

    def _require_territory(self, tid: str) -> None:
        if tid not in self.G:
            raise TerritoryError(f"Territorio desconocido: '{tid}'")

    def _require_neighbor(self, src: str, dst: str) -> None:
        if not self.G.has_edge(src, dst):
            raise NotNeighborError(
                f"'{src}' y '{dst}' no son territorios vecinos."
            )

    def _require_owner(self, tid: str, player_id: str) -> None:
        if self.G.nodes[tid]["owner"] != player_id:
            raise NotOwnerError(
                f"El jugador '{player_id}' no controla '{tid}'."
            )

    def _require_troops(self, tid: str, amount: int) -> None:
        current = self.G.nodes[tid]["troops"]
        if current < amount:
            raise InsufficientTroopsError(
                f"'{tid}' solo tiene {current} tropas; se necesitan {amount}."
            )

    # ── Lectura de atributos ──────────────────────────────────

    def get(self, tid: str) -> dict:
        """Devuelve todos los atributos de un territorio."""
        self._require_territory(tid)
        return dict(self.G.nodes[tid])

    def owner(self, tid: str) -> Optional[str]:
        return self.G.nodes[tid]["owner"]

    def troops(self, tid: str) -> int:
        return self.G.nodes[tid]["troops"]

    def continent(self, tid: str) -> str:
        return self.G.nodes[tid]["continent"]

    def all_territories(self) -> list[str]:
        return list(self.G.nodes)

    # ── Vecinos ───────────────────────────────────────────────

    def neighbors(self, tid: str) -> list[str]:
        """Devuelve los territorios adyacentes a tid."""
        self._require_territory(tid)
        return list(self.G.neighbors(tid))

    def are_neighbors(self, a: str, b: str) -> bool:
        """Comprueba si dos territorios comparten frontera."""
        return self.G.has_edge(a, b)

    def enemy_neighbors(self, tid: str, player_id: str) -> list[str]:
        """Vecinos que pertenecen a otro jugador (objetivos de ataque)."""
        return [
            n for n in self.neighbors(tid)
            if self.G.nodes[n]["owner"] != player_id
        ]

    def friendly_neighbors(self, tid: str, player_id: str) -> list[str]:
        """Vecinos que pertenecen al mismo jugador."""
        return [
            n for n in self.neighbors(tid)
            if self.G.nodes[n]["owner"] == player_id
        ]

    # ── Agregar tropas ────────────────────────────────────────

    def add_troops(self, tid: str, amount: int, player_id: str) -> int:
        """
        Coloca `amount` tropas en un territorio propio.
        Devuelve el nuevo total de tropas.
        """
        self._require_territory(tid)
        self._require_owner(tid, player_id)
        if amount <= 0:
            raise TerritoryError("La cantidad de tropas debe ser positiva.")

        self.G.nodes[tid]["troops"] += amount
        return self.G.nodes[tid]["troops"]

    # ── Mover tropas ──────────────────────────────────────────

    def move_troops(
        self,
        src: str,
        dst: str,
        amount: int,
        player_id: str,
    ) -> None:
        """
        Mueve `amount` tropas de src a dst.
        Ambos territorios deben ser del mismo jugador y ser vecinos.
        Se debe dejar al menos 1 tropa en el origen.
        """
        self._require_territory(src)
        self._require_territory(dst)
        self._require_neighbor(src, dst)
        self._require_owner(src, player_id)
        self._require_owner(dst, player_id)
        self._require_troops(src, amount + 1)   # deja al menos 1

        self.G.nodes[src]["troops"] -= amount
        self.G.nodes[dst]["troops"] += amount

    # ── Conquista ─────────────────────────────────────────────

    def conquer(
        self,
        attacker_tid: str,
        defender_tid: str,
        attacker_id: str,
        troops_to_move: int,
    ) -> None:
        """
        Registra la conquista de defender_tid por attacker_id.

        Se asume que CombatResolver ya determinó que el defensor
        perdió todas sus tropas. Este método actualiza el grafo:
          - Cambia el owner del territorio conquistado
          - Establece las tropas que se mueven al nuevo territorio
          - Deja 1 tropa en el atacante (mínimo reglamentario)

        Args:
            attacker_tid   : territorio desde donde se atacó
            defender_tid   : territorio conquistado
            attacker_id    : jugador que conquista
            troops_to_move : tropas que ocupan el nuevo territorio
        """
        self._require_territory(attacker_tid)
        self._require_territory(defender_tid)
        self._require_neighbor(attacker_tid, defender_tid)
        self._require_owner(attacker_tid, attacker_id)
        self._require_troops(attacker_tid, troops_to_move + 1)

        self.G.nodes[defender_tid]["owner"] = attacker_id
        self.G.nodes[defender_tid]["troops"] = troops_to_move
        self.G.nodes[attacker_tid]["troops"] -= troops_to_move

    # ── Análisis del grafo ────────────────────────────────────

    def territories_by_player(self, player_id: str) -> list[str]:
        """Lista de todos los territorios que controla un jugador."""
        return [
            n for n in self.G.nodes
            if self.G.nodes[n]["owner"] == player_id
        ]

    def controls_continent(self, player_id: str, continent: str) -> bool:
        """Devuelve True si el jugador controla todos los territorios del continente."""
        continent_nodes = [
            n for n in self.G.nodes
            if self.G.nodes[n]["continent"] == continent
        ]
        return all(
            self.G.nodes[n]["owner"] == player_id
            for n in continent_nodes
        )

    def controlled_continents(self, player_id: str) -> list[str]:
        """Lista de continentes completamente controlados por el jugador."""
        all_continents = set(nx.get_node_attributes(self.G, "continent").values())
        return [c for c in all_continents if self.controls_continent(player_id, c)]

    def shortest_path(self, src: str, dst: str) -> list[str]:
        """Ruta más corta entre dos territorios (solo por el grafo, sin restricción de dueño)."""
        try:
            return nx.shortest_path(self.G, src, dst)
        except nx.NetworkXNoPath:
            return []

    def player_is_connected(self, player_id: str) -> bool:
        """
        Verifica si todos los territorios del jugador forman
        un componente conexo (útil para estrategia de IA).
        """
        owned = self.territories_by_player(player_id)
        if len(owned) <= 1:
            return True
        subgraph = self.G.subgraph(owned)
        return nx.is_connected(subgraph)

    def frontier_territories(self, player_id: str) -> list[str]:
        """Territorios propios que tienen al menos un vecino enemigo."""
        return [
            t for t in self.territories_by_player(player_id)
            if self.enemy_neighbors(t, player_id)
        ]

    # ── Inicialización de partida ─────────────────────────────

    def assign_territory(self, tid: str, player_id: str, troops: int = 1) -> None:
        """Asigna un territorio a un jugador al inicio de la partida."""
        self._require_territory(tid)
        self.G.nodes[tid]["owner"] = player_id
        self.G.nodes[tid]["troops"] = troops

    def summary(self) -> dict:
        """Resumen del estado actual del grafo."""
        owners: dict[str, list] = {}
        for n, data in self.G.nodes(data=True):
            owner = data["owner"] or "Sin dueño"
            owners.setdefault(owner, []).append(n)

        return {
            "total_territories": self.G.number_of_nodes(),
            "total_borders": self.G.number_of_edges(),
            "by_player": owners,
        }

    def __repr__(self) -> str:
        return (
            f"WorldGraph("
            f"{self.G.number_of_nodes()} territorios, "
            f"{self.G.number_of_edges()} fronteras)"
        )
