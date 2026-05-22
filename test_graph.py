"""
test_graph.py
Prueba completa del sistema de grafos.
Ejecutar desde la raíz del proyecto: python test_graph.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.world_graph import WorldGraph, TerritoryError, NotNeighborError, InsufficientTroopsError
from graph.graph_visualizer import draw_map


def separador(titulo: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {titulo}")
    print(f"{'─' * 55}")


def test_construccion(world: WorldGraph) -> None:
    separador("1. Construcción del grafo")
    print(world)
    s = world.summary()
    print(f"  Territorios totales : {s['total_territories']}")
    print(f"  Fronteras totales   : {s['total_borders']}")
    print(f"  Sin dueño           : {len(s['by_player'].get('Sin dueño', []))}")


def test_vecinos(world: WorldGraph) -> None:
    separador("2. Vecinos")
    for tid in ["CAN", "BRA", "NAF"]:
        vecinos = world.neighbors(tid)
        print(f"  {tid:4} → vecinos: {vecinos}")
    print()
    print(f"  ¿CAN-USA vecinos? {world.are_neighbors('CAN', 'USA')}")
    print(f"  ¿CAN-BRA vecinos? {world.are_neighbors('CAN', 'BRA')}")
    print(f"  ¿BRA-NAF vecinos? {world.are_neighbors('BRA', 'NAF')}")


def test_asignacion(world: WorldGraph) -> None:
    separador("3. Asignación de territorios (inicio de partida)")

    # Player 1 controla N. América
    na_territories = ["CAN", "USA", "MEX", "CAR", "GRN", "CAM"]
    for tid in na_territories:
        world.assign_territory(tid, "player1", troops=3)

    # Player 2 controla S. América
    sa_territories = ["VEN", "COL", "BRA", "ARG", "PER", "CHI"]
    for tid in sa_territories:
        world.assign_territory(tid, "player2", troops=2)

    # Player 3 controla África
    af_territories = ["NAF", "EGY", "WAF", "CAF", "EAF", "SAF", "MAD"]
    for tid in af_territories:
        world.assign_territory(tid, "player3", troops=2)

    s = world.summary()
    for player, territories in s["by_player"].items():
        print(f"  {player:10} → {territories}")


def test_agregar_tropas(world: WorldGraph) -> None:
    separador("4. Agregar tropas")
    before = world.troops("CAN")
    world.add_troops("CAN", 4, "player1")
    after  = world.troops("CAN")
    print(f"  CAN: {before}T → {after}T (+4)")

    # Error: jugador incorrecto
    try:
        world.add_troops("CAN", 2, "player2")
    except TerritoryError as e:
        print(f"  [OK] Error esperado: {e}")


def test_mover_tropas(world: WorldGraph) -> None:
    separador("5. Mover tropas entre territorios propios")
    print(f"  CAN antes: {world.troops('CAN')}T  |  USA antes: {world.troops('USA')}T")
    world.move_troops("CAN", "USA", 3, "player1")
    print(f"  CAN  después: {world.troops('CAN')}T  |  USA después: {world.troops('USA')}T")

    # Error: no vecinos
    try:
        world.move_troops("CAN", "MEX", 1, "player1")   # CAN y MEX no son vecinos
    except NotNeighborError as e:
        print(f"  [OK] Error esperado: {e}")

    # Error: tropas insuficientes
    try:
        world.move_troops("CAN", "USA", 100, "player1")
    except InsufficientTroopsError as e:
        print(f"  [OK] Error esperado: {e}")


def test_conquistar(world: WorldGraph) -> None:
    separador("6. Conquistar territorio")
    # player1 ataca VEN desde CAR (son vecinos, CAR=player1, VEN=player2)
    print(f"  CAR antes  : dueño={world.owner('CAR')}, tropas={world.troops('CAR')}")
    print(f"  VEN antes  : dueño={world.owner('VEN')}, tropas={world.troops('VEN')}")

    # Simula que el combate terminó: player1 conquista VEN
    world.G.nodes["VEN"]["troops"] = 0   # defensor derrotado
    world.conquer("CAR", "VEN", "player1", troops_to_move=2)

    print(f"  CAR después: dueño={world.owner('CAR')}, tropas={world.troops('CAR')}")
    print(f"  VEN después: dueño={world.owner('VEN')}, tropas={world.troops('VEN')}")


def test_analisis(world: WorldGraph) -> None:
    separador("7. Análisis del grafo")

    # Continentes controlados
    for pid in ["player1", "player2", "player3"]:
        continentes = world.controlled_continents(pid)
        print(f"  {pid} controla continentes: {continentes or 'ninguno'}")

    # Territorios en frontera
    fronteras = world.frontier_territories("player1")
    print(f"\n  Fronteras de player1: {fronteras}")

    # Conectividad
    for pid in ["player1", "player2", "player3"]:
        conectado = world.player_is_connected(pid)
        print(f"  {pid} territorios conectados: {conectado}")

    # Ruta más corta
    ruta = world.shortest_path("CAN", "SAF")
    print(f"\n  Ruta CAN → SAF: {' → '.join(ruta)}")

    # Vecinos enemigos desde USA
    enemigos = world.enemy_neighbors("USA", "player1")
    print(f"  Vecinos enemigos de USA (player1): {enemigos}")


def test_visualizacion(world: WorldGraph) -> None:
    separador("8. Generando visualización")
    path = draw_map(world, output_path="output/risk_map.png")
    print(f"  Mapa guardado en: {path}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    world = WorldGraph()

    test_construccion(world)
    test_vecinos(world)
    test_asignacion(world)
    test_agregar_tropas(world)
    test_mover_tropas(world)
    test_conquistar(world)
    test_analisis(world)
    test_visualizacion(world)

    print("\n  ✓ Todas las pruebas completadas.\n")
