"""
test_engine.py
Prueba del motor de juego y la IA con cadenas de Markov.
Ejecutar desde la raíz: python test_engine.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from graph.world_graph import WorldGraph
from simulation.markov_ai import MarkovAI, STATES, BASE_TRANSITION
from game.engine import GameEngine, PlayerInfo
from game.state import StateManager


def sep(t): print(f"\n{'─'*55}\n  {t}\n{'─'*55}")


def make_engine(seed=42):
    world = WorldGraph()
    ai1 = MarkovAI("player1", rng=np.random.default_rng(1))
    ai2 = MarkovAI("player2", rng=np.random.default_rng(2))
    ai3 = MarkovAI("player3", rng=np.random.default_rng(3))
    players = [
        PlayerInfo("player1", "Rojo",  "#E05A3A", is_ai=True,  ai_agent=ai1),
        PlayerInfo("player2", "Azul",  "#4A90D9", is_ai=True,  ai_agent=ai2),
        PlayerInfo("player3", "Verde", "#6DBF6D", is_ai=True,  ai_agent=ai3),
    ]
    return GameEngine(world, players, seed=seed)


def test_markov_chain():
    sep("1. Cadena de Markov — distribución estacionaria")
    ai = MarkovAI("test")
    stat = ai.stationary_distribution()
    print(f"  Matriz de transición P:")
    for i, row in enumerate(BASE_TRANSITION):
        print(f"    {STATES[i]:12} → {[f'{v:.2f}' for v in row]}")
    print(f"\n  Distribución estacionaria π:")
    for i, s in enumerate(STATES):
        bar = "█" * int(stat[i] * 40)
        print(f"    {s:12} {bar} {stat[i]:.3f}")
    print(f"\n  Suma π = {stat.sum():.6f} (debe ser 1.0)")


def test_setup():
    sep("2. Setup inicial de partida")
    engine = make_engine()
    engine.setup()
    s = engine.get_state()
    print(f"  Turno: {s['turn']} | Fase: {s['phase']}")
    for p in s["players"]:
        print(f"  {p['name']:8} | {len(p['territories']):2} terr. | "
              f"{p['troops_total']:3} tropas | +{p['reinforcements']} refuerzos/turno")


def test_reinforcements():
    sep("3. Cálculo de refuerzos")
    engine = make_engine()
    engine.setup()
    for p in engine.active_players:
        r = engine.calculate_reinforcements(p.player_id)
        conts = engine.world.controlled_continents(p.player_id)
        owned = len(engine.world.territories_by_player(p.player_id))
        print(f"  {p.name:8} | {owned} terr. → base={max(3,owned//3)} "
              f"+ bonus={r - max(3,owned//3)} (conts={conts or '—'}) = {r}")


def test_ai_decisions():
    sep("4. Decisiones de la IA (un turno manual)")
    engine = make_engine(seed=10)
    engine.setup()

    p = engine.players[0]
    ai = p.ai_agent
    world = engine.world

    # Forzar estado de expansión para ver decisiones agresivas
    ai.current_state = 1   # EXPANSION

    troops = engine.calculate_reinforcements(p.player_id)
    print(f"\n  {p.name} en estado: {ai.state_name}")
    print(f"  Tropas disponibles: {troops}")

    refuerzos = ai.decide_reinforce(world, troops)
    print(f"\n  Top 3 refuerzos:")
    for a in refuerzos[:3]:
        print(f"    {a}")

    ataques = ai.decide_attacks(world)
    print(f"\n  Top 3 ataques posibles:")
    for a in ataques[:3]:
        print(f"    {a}")


def test_full_game():
    sep("5. Partida completa IA vs IA vs IA")
    engine = make_engine(seed=99)
    winner = engine.run_full_game(max_turns=30, verbose=True)

    sep("  Resultado final")
    for p in engine.players:
        owned  = engine.world.territories_by_player(p.player_id)
        troops = sum(engine.world.troops(t) for t in owned)
        status = "GANADOR 🏆" if p.player_id == winner else ("eliminado" if p.eliminated else "activo")
        print(f"  {p.name:8} | {len(owned):2} terr. | {troops:3} tropas | {status}")
        if p.ai_agent:
            print(p.ai_agent.state_report())


def test_save_load():
    sep("6. Guardar y cargar partida")
    engine = make_engine(seed=5)
    engine.setup()
    engine.turn = 3

    path = StateManager.save(engine, "saves/test_save.json")
    print(f"  Partida guardada en: {path}")

    data = StateManager.load(path)
    print(f"  Cargado: turno={data['turn']}, fase={data['phase']}, "
          f"territorios={len(data['territories'])}")
    print(f"  Guardado el: {data['saved_at']}")


def test_state_api():
    sep("7. Estado serializable (para Flask)")
    engine = make_engine()
    engine.setup()
    state = engine.get_state()

    print(f"  Claves del estado: {list(state.keys())}")
    print(f"  Territorios: {len(state['territories'])}")
    print(f"  Fronteras: {len(state['borders'])}")
    print(f"  Jugadores: {len(state['players'])}")

    # Muestra un territorio de ejemplo
    tid = "BRA"
    t = state["territories"][tid]
    print(f"\n  Ejemplo territorio BRA:")
    for k, v in t.items():
        print(f"    {k:12}: {v}")


if __name__ == "__main__":
    test_markov_chain()
    test_setup()
    test_reinforcements()
    test_ai_decisions()
    test_full_game()
    test_save_load()
    test_state_api()
    print("\n  ✓ Todas las pruebas del motor completadas.\n")
