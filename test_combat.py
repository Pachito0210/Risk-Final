"""
test_combat.py
Prueba completa del sistema de combate.
Ejecutar desde la raíz: python test_combat.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from simulation.combat import (
    CombatResolver, MarkovCombat, CombatSimulator, get_round_probs
)
from graph.world_graph import WorldGraph


def sep(titulo):
    print(f"\n{'─'*55}\n  {titulo}\n{'─'*55}")


# ── 1. Probabilidades exactas de una ronda ────────────────────
def test_round_probs():
    sep("1. Probabilidades exactas por ronda")
    casos = [(3,2), (2,2), (1,2), (3,1), (2,1), (1,1)]
    for n_atk, n_def in casos:
        probs = get_round_probs(n_atk + 1, n_def)  # +1 porque internamente usa atk-1
        print(f"\n  {n_atk} dados ATK vs {n_def} dados DEF:")
        for (al, dl), p in sorted(probs.items()):
            print(f"    ATK pierde {al}, DEF pierde {dl}  →  p = {p:.4f}")


# ── 2. Ronda individual ───────────────────────────────────────
def test_single_round():
    sep("2. Ronda individual de dados")
    resolver = CombatResolver(np.random.default_rng(42))
    for _ in range(5):
        r = resolver.resolve_round(atk_troops=4, def_troops=3)
        print(f"  {r}")


# ── 3. Batalla completa con log ───────────────────────────────
def test_full_battle():
    sep("3. Batalla completa con log de rondas")
    resolver = CombatResolver(np.random.default_rng(7))
    result = resolver.resolve_battle(5, 3, log_rounds=True)
    print(f"  {result.summary()}")
    print(f"  Tasa pérdida ATK: {result.atk_loss_rate:.0%}")
    print(f"  Tasa pérdida DEF: {result.def_loss_rate:.0%}")
    print("\n  Rondas:")
    for i, r in enumerate(result.round_log, 1):
        print(f"    {i:2}. {r}")


# ── 4. Probabilidad analítica Markov ──────────────────────────
def test_markov_probs():
    sep("4. Probabilidades analíticas (cadena de Markov)")
    markov = MarkovCombat()
    print(f"  {'ATK':>4}  {'DEF':>4}  {'P(atk gana)':>12}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*12}")
    combos = [(2,1),(3,2),(4,3),(5,3),(5,5),(8,3),(3,8),(10,2)]
    for a, d in combos:
        p = markov.win_probability(a, d)
        bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
        print(f"  {a:>4}  {d:>4}  {p:>10.1%}  {bar}")


# ── 5. Pérdidas esperadas ─────────────────────────────────────
def test_expected_losses():
    sep("5. Pérdidas esperadas (distribución completa de estados)")
    markov = MarkovCombat()
    for a, d in [(5, 3), (3, 5), (4, 4)]:
        stats = markov.expected_losses(a, d)
        print(f"\n  {a} ATK vs {d} DEF:")
        print(f"    P(atk gana)       = {stats['win_prob']:.1%}")
        print(f"    Pérdida esperada ATK = {stats['atk_expected_loss']:.2f} tropas")
        print(f"    Pérdida esperada DEF = {stats['def_expected_loss']:.2f} tropas")


# ── 6. Matriz de transición numpy ────────────────────────────
def test_transition_matrix():
    sep("6. Matriz de transición P (cadena de Markov)")
    markov = MarkovCombat()
    P, states, idx = markov.transition_matrix(atk_troops=3, def_troops=2)
    print(f"  Estados: {states}")
    print(f"  Dimensión de P: {P.shape}")
    print(f"\n  Matriz P (filas = origen, columnas = destino):")
    print(f"  {'':8}", end="")
    for s in states:
        print(f"  {str(s):>8}", end="")
    print()
    for i, s in enumerate(states):
        print(f"  {str(s):>8}", end="")
        for j in range(len(states)):
            v = P[i][j]
            print(f"  {v:>8.3f}" if v > 0 else f"  {'·':>8}", end="")
        print()
    print(f"\n  Verificación filas suman 1: {np.allclose(P.sum(axis=1), 1.0)}")


# ── 7. Simulación Monte Carlo ─────────────────────────────────
def test_monte_carlo():
    sep("7. Simulación Monte Carlo (5 000 batallas)")
    sim = CombatSimulator(seed=0)
    for a, d in [(5, 3), (3, 5), (4, 4)]:
        stats = sim.run(a, d, n=5000)
        print(stats)


# ── 8. Tabla de probabilidades ───────────────────────────────
def test_probability_table():
    sep("8. Tabla P(atk gana) — tropas ATK vs DEF")
    sim = CombatSimulator()
    table = sim.probability_table(max_atk=8, max_def=6)

    # Encabezado
    header = "ATK\\DEF"
    print(f"\n  {header:>8}", end="")
    for d in range(1, 7):
        print(f"  DEF={d}", end="")
    print()
    print(f"  {'─'*8}", end="")
    for _ in range(6):
        print(f"  {'─'*5}", end="")
    print()

    for i, a in enumerate(range(2, 9)):
        print(f"  ATK={a:>4} ", end="")
        for j in range(6):
            v = table[i][j]
            print(f"  {v:.2f} ", end="")
        print()


# ── 9. Integración con WorldGraph ────────────────────────────
def test_with_worldgraph():
    sep("9. Integración con WorldGraph")
    world = WorldGraph()

    # Setup: player1 en N.América, player2 en S.América
    na = ["CAN", "USA", "MEX", "CAR", "GRN", "CAM"]
    sa = ["VEN", "COL", "BRA", "ARG", "PER", "CHI"]
    for tid in na:
        world.assign_territory(tid, "player1", troops=5)
    for tid in sa:
        world.assign_territory(tid, "player2", troops=3)

    print(f"\n  Estado antes del ataque:")
    print(f"    CAR → dueño={world.owner('CAR')}, tropas={world.troops('CAR')}")
    print(f"    VEN → dueño={world.owner('VEN')}, tropas={world.troops('VEN')}")

    sim = CombatResolver(np.random.default_rng(99))
    result = sim.attack_territory(
        world,
        from_tid="CAR",
        to_tid="VEN",
        attacker_id="player1",
        troops_to_move=2,
        log_rounds=True,
    )

    print(f"\n  Resultado: {result.summary()}")
    print(f"\n  Estado después del ataque:")
    print(f"    CAR → dueño={world.owner('CAR')}, tropas={world.troops('CAR')}")
    print(f"    VEN → dueño={world.owner('VEN')}, tropas={world.troops('VEN')}")


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_round_probs()
    test_single_round()
    test_full_battle()
    test_markov_probs()
    test_expected_losses()
    test_transition_matrix()
    test_monte_carlo()
    test_probability_table()
    test_with_worldgraph()
    print("\n  ✓ Todas las pruebas de combate completadas.\n")
