"""
flask_app.py
Servidor Flask que expone el GameEngine como API REST.
El frontend HTML/JS consume estos endpoints.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template
import numpy as np
from graph.world_graph import WorldGraph, TerritoryError
from simulation.combat import CombatResolver, MarkovCombat
from simulation.markov_ai import MarkovAI
from game.engine import GameEngine, PlayerInfo
from game.state import StateManager

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Estado global del juego ───────────────────────────────────
_engine: GameEngine | None = None


def get_engine() -> GameEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Partida no iniciada")
    return _engine


def _build_ai_summary(engine, ai_logs):
    """Convierte los logs internos de la IA en un resumen serializable para el frontend."""
    ai_summary = []
    for entry in ai_logs:
        log     = entry["log"]
        attacks = log.get("attacks", [])

        reinforce_detail = [
            {"territory": a.target, "troops": a.troops,
             "name": engine.world.G.nodes[a.target].get("name", a.target)}
            for a in log.get("reinforce", [])
        ]

        attack_detail = []
        for action, result in attacks:
            from_name = engine.world.G.nodes[action.source].get("name", action.source)
            to_name   = engine.world.G.nodes[action.target].get("name", action.target)
            attack_detail.append({
                "from":      action.source,
                "to":        action.target,
                "from_name": from_name,
                "to_name":   to_name,
                "won":       result.attacker_won,
                "atk_start": result.atk_start,
                "def_start": result.def_start,
                "atk_rem":   result.atk_remaining,
                "def_rem":   result.def_remaining,
            })

        fortify = log.get("fortify")
        fortify_detail = None
        if fortify:
            fortify_detail = {
                "from":      fortify.source,
                "to":        fortify.target,
                "from_name": engine.world.G.nodes[fortify.source].get("name", fortify.source),
                "to_name":   engine.world.G.nodes[fortify.target].get("name", fortify.target),
                "troops":    fortify.troops,
            }

        ai_summary.append({
            "player":    entry["player"],
            "name":      entry["name"],
            "color":     entry["color"],
            "state":     log.get("state", "?"),
            "conquered": log.get("conquered", []),
            "reinforce": reinforce_detail,
            "attacks":   attack_detail,
            "fortify":   fortify_detail,
        })
    return ai_summary


# ─────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    """Inicia una nueva partida. Body JSON: { player_name, n_ai }"""
    global _engine
    data        = request.get_json(force=True) or {}
    player_name = data.get("player_name", "Tú")
    n_ai        = min(int(data.get("n_ai", 2)), 3)
    seed        = data.get("seed", None)

    world = WorldGraph()

    ai_colors = ["#E05A3A", "#6DBF6D", "#F5A623"]
    ai_names  = ["IA Roja", "IA Verde", "IA Naranja"]

    players = [
        PlayerInfo("human", player_name, "#4A90D9", is_ai=False)
    ]
    for i in range(n_ai):
        ai_id = f"ai{i+1}"
        agent = MarkovAI(ai_id, rng=np.random.default_rng(i * 17 + 42))
        players.append(PlayerInfo(
            ai_id, ai_names[i], ai_colors[i],
            is_ai=True, ai_agent=agent
        ))

    _engine = GameEngine(world, players, seed=seed)
    _engine.setup()

    return jsonify({"ok": True, "state": _engine.get_state()})


@app.route("/api/state")
def get_state():
    """Devuelve el estado actual del juego."""
    try:
        return jsonify(_engine.get_state())
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/reinforce", methods=["POST"])
def reinforce():
    """
    Coloca tropas en un territorio del jugador humano.
    Body: { territory_id, troops }
    """
    engine = get_engine()
    if engine.current_player.player_id != "human":
        return jsonify({"error": "No es tu turno"}), 400
    if engine.phase != "draft":
        return jsonify({"error": "No estás en fase de refuerzo"}), 400

    data   = request.get_json(force=True)
    tid    = data.get("territory_id")
    troops = int(data.get("troops", 1))

    try:
        result = engine.human_reinforce(tid, troops)
        return jsonify({**result, "state": engine.get_state()})
    except TerritoryError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/attack", methods=["POST"])
def attack():
    engine = get_engine()
    if engine.current_player.player_id != "human":
        return jsonify({"error": "No es tu turno"}), 400
    if engine.phase != "attack":
        return jsonify({"error": "No estas en fase de ataque"}), 400
    data           = request.get_json(force=True)
    from_tid       = data.get("from_tid")
    to_tid         = data.get("to_tid")
    troops_to_move = int(data.get("troops_to_move", 1))
    try:
        result = engine.human_attack(from_tid, to_tid, troops_to_move)
        return jsonify({**result, "state": engine.get_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/roll_dice", methods=["POST"])
def roll_dice():
    """Lanza una ronda de dados sin modificar el mapa."""
    engine = get_engine()
    if engine.current_player.player_id != "human":
        return jsonify({"error": "No es tu turno"}), 400
    data     = request.get_json(force=True)
    from_tid = data.get("from_tid")
    to_tid   = data.get("to_tid")
    atk_dice = int(data.get("atk_dice", 1))
    t_atk = engine.world.troops(from_tid)
    t_def = engine.world.troops(to_tid)
    max_atk = min(t_atk - 1, 3)
    max_def = min(t_def, 2)
    atk_dice = max(1, min(atk_dice, max_atk))
    from simulation.combat import CombatResolver, MarkovCombat
    resolver = CombatResolver()
    atk_rolls = resolver.roll_dice(atk_dice)
    def_rolls = resolver.roll_dice(max_def)
    atk_loss = def_loss = 0
    for a, d in zip(atk_rolls, def_rolls):
        if a > d:
            def_loss += 1
        else:
            atk_loss += 1
    mc = MarkovCombat()
    win_prob = mc.win_probability(t_atk, t_def)
    return jsonify({
        "atk_rolls": atk_rolls, "def_rolls": def_rolls,
        "atk_loss": atk_loss, "def_loss": def_loss,
        "atk_troops": t_atk, "def_troops": t_def,
        "max_atk_dice": max_atk, "max_def_dice": max_def,
        "win_prob": round(win_prob, 4),
    })


@app.route("/api/apply_combat", methods=["POST"])
def apply_combat():
    """Aplica el resultado acumulado de un combate interactivo."""
    engine = get_engine()
    if engine.current_player.player_id != "human":
        return jsonify({"error": "No es tu turno"}), 400
    data           = request.get_json(force=True)
    from_tid       = data.get("from_tid")
    to_tid         = data.get("to_tid")
    atk_loss_total = int(data.get("atk_loss_total", 0))
    def_loss_total = int(data.get("def_loss_total", 0))
    troops_to_move = int(data.get("troops_to_move", 1))
    defender_owner = data.get("defender_owner", "")
    world = engine.world
    atk_troops = world.troops(from_tid)
    def_troops = world.troops(to_tid)
    world.G.nodes[from_tid]["troops"] = max(1, atk_troops - atk_loss_total)
    world.G.nodes[to_tid]["troops"]   = max(0, def_troops - def_loss_total)
    attacker_won = world.troops(to_tid) == 0
    if attacker_won:
        final_atk = world.troops(from_tid)
        move = max(1, min(troops_to_move, final_atk - 1))
        world.G.nodes[to_tid]["owner"]    = "human"
        world.G.nodes[to_tid]["troops"]   = move
        world.G.nodes[from_tid]["troops"] = final_atk - move
        engine._check_elimination(defender_owner, to_tid)
    engine._check_winner()
    engine._log("attack","human","interactive_attack",f"{from_tid}->{to_tid} won={attacker_won}")
    return jsonify({"ok": True, "attacker_won": attacker_won, "state": engine.get_state()})


@app.route("/api/fortify", methods=["POST"])
def fortify():
    """
    Mueve tropas entre territorios propios.
    Body: { from_tid, to_tid, troops }
    """
    engine = get_engine()
    if engine.current_player.player_id != "human":
        return jsonify({"error": "No es tu turno"}), 400
    if engine.phase != "fortify":
        return jsonify({"error": "No estás en fase de fortificación"}), 400

    data   = request.get_json(force=True)
    try:
        result = engine.human_fortify(
            data["from_tid"], data["to_tid"], int(data.get("troops", 1))
        )
        return jsonify({**result, "state": engine.get_state()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/end_phase", methods=["POST"])
def end_phase():
    """
    Termina la fase actual del jugador humano.
    - Si hay fases restantes en el turno humano (draft->attack->fortify), avanza la fase.
    - Si el humano termina fortify, pasa a las IAs y ejecuta TODOS sus turnos completos
      (draft+attack+fortify cada una) antes de devolver el control al humano.
    """
    engine = get_engine()

    # Sincronizar: si el turno ya paso a la IA (por ejemplo doble click),
    # devolver el estado actual sin error para que el frontend se sincronice
    if engine.current_player.player_id != "human":
        return jsonify({
            "phase": engine.phase,
            "player": engine.current_player.player_id,
            "state": engine.get_state(),
            "ai_logs": []
        })

    # Avanzar la fase del humano
    engine.phase_idx += 1
    if engine.phase_idx >= 3:  # despues de fortify
        engine.phase_idx = 0
        engine._next_player()

    ai_logs = []

    # Si ahora le toca a las IAs, ejecutar TODOS sus turnos completos en secuencia
    if not engine.winner and engine.current_player.player_id != "human":
        safety = 0
        while not engine.winner and engine.current_player.player_id != "human":
            safety += 1
            if safety > 50:
                break
            p = engine.current_player
            if p.eliminated:
                engine._next_player()
                continue
            if p.is_ai and p.ai_agent:
                troops = engine.calculate_reinforcements(p.player_id)
                log = p.ai_agent.take_turn(engine.world, troops, verbose=False)
                engine._log("full_turn", p.player_id, "ai_turn",
                            f"estado={log['state']}, conquistas={log['conquered']}")
                engine._check_winner()
                ai_logs.append({
                    "player": p.player_id,
                    "name":   p.name,
                    "color":  p.color,
                    "log":    log,
                })
                engine._next_player()
            else:
                break

    ai_summary = _build_ai_summary(engine, ai_logs)

    return jsonify({
        "phase":   engine.phase,
        "player":  engine.current_player.player_id,
        "state":   engine.get_state(),
        "ai_logs": ai_summary,
    })


@app.route("/api/win_prob", methods=["GET"])
def win_prob():
    """
    Calcula P(atacante gana) para una combinación de tropas.
    Query params: atk, def
    """
    atk = int(request.args.get("atk", 2))
    def_ = int(request.args.get("def", 1))
    mc  = MarkovCombat()
    p   = mc.win_probability(atk, def_)
    exp = mc.expected_losses(atk, def_)
    return jsonify({
        "win_prob":       round(p, 4),
        "atk_exp_loss":   round(exp["atk_expected_loss"], 2),
        "def_exp_loss":   round(exp["def_expected_loss"], 2),
    })


@app.route("/api/save", methods=["POST"])
def save():
    engine = get_engine()
    path   = StateManager.save(engine, "saves/partida.json")
    return jsonify({"ok": True, "path": path})


@app.route("/api/reinforcements_available")
def reinforcements_available():
    """Tropas disponibles para el jugador humano este turno."""
    engine = get_engine()
    n = engine.calculate_reinforcements("human")
    # Restar las ya colocadas en este turno
    placed = 0
    for e in engine.events:
        if e.turn == engine.turn and e.phase == "draft" and e.player == "human" and "+" in e.details:
            try:
                placed += int(e.details.split("+")[1].split(" ")[0])
            except Exception:
                pass
    return jsonify({"available": max(0, n - placed), "total": n})



@app.route("/mapper")
def mapper():
    return render_template("mapper.html")

@app.route("/api/map_coords")
def map_coords():
    """Coordenadas de territorios como % del mapa."""
    coords = {
        "GRN": [52.5, 10.0], "CAN": [40.0, 20.0], "USA": [35.0, 30.0],
        "MEX": [30.0, 42.0], "CAR": [42.0, 44.0], "CAM": [30.0, 50.0],
        "VEN": [38.0, 56.0], "COL": [34.0, 62.0], "BRA": [43.0, 65.0],
        "PER": [32.0, 70.0], "ARG": [36.0, 78.0], "CHI": [31.0, 80.0],
        "NAF": [60.0, 42.0], "EGY": [65.0, 38.0], "WAF": [57.0, 54.0],
        "CAF": [63.0, 56.0], "EAF": [70.0, 52.0], "SAF": [64.0, 72.0],
        "MAD": [72.0, 70.0],
    }
    return jsonify(coords)

def create_app():
    return app


if __name__ == "__main__":
    app.run(debug=True, port=5000)
