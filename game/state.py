"""
state.py
Guardado y carga de partidas en JSON.
"""

import json, os
from datetime import datetime
from game.engine import GameEngine


class StateManager:

    @staticmethod
    def save(engine: GameEngine, path: str = "saves/partida.json") -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = engine.get_state()
        state["saved_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def list_saves(folder: str = "saves") -> list[str]:
        if not os.path.exists(folder):
            return []
        return sorted(
            f for f in os.listdir(folder) if f.endswith(".json")
        )
