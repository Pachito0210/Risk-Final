"""
main.py
Punto de entrada del juego Risk.
Ejecutar desde la raíz del proyecto: python main.py
Luego abrir: http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend.flask_app import app

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  RISK — Teoría de Grafos · Markov · Probabilidades")
    print("="*50)
    print("  Abre tu navegador en: http://localhost:5000")
    print("  Presiona Ctrl+C para salir.")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
