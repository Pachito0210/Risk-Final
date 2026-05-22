# 🎲 Risk Web

Versión web del juego **Risk** con IA basada en cadenas de Markov.  
Jugable desde cualquier navegador, sin instalar nada.

---

## 🚀 Deploy en Railway (gratis, 5 minutos)

Railway es la forma más rápida de tener el juego online. Tiene plan gratuito.

### Paso 1 — Subir el código a GitHub

1. Ve a [github.com](https://github.com) e inicia sesión (o crea cuenta gratis).
2. Haz clic en **"New repository"** (botón verde arriba a la derecha).
3. Ponle nombre: `risk-web`, déjalo en **Public**, y clic en **"Create repository"**.
4. En tu computador, abre una terminal en la carpeta del proyecto y ejecuta:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/risk-web.git
git push -u origin main
```

> Reemplaza `TU_USUARIO` con tu nombre de usuario de GitHub.

---

### Paso 2 — Deploy en Railway

1. Ve a [railway.app](https://railway.app) y entra con tu cuenta de GitHub.
2. Haz clic en **"New Project"**.
3. Selecciona **"Deploy from GitHub repo"**.
4. Elige el repositorio `risk-web`.
5. Railway detecta automáticamente la configuración y empieza a construir.
6. Espera ~2 minutos mientras termina el build.
7. Haz clic en **"Generate Domain"** (o Settings → Networking → Generate Domain).
8. ¡Listo! Recibirás una URL tipo `risk-web-production.up.railway.app`.

Comparte esa URL y cualquiera puede jugar desde su navegador. 🎮

---

## 🕹️ Cómo jugar

1. Abre la URL del juego en tu navegador.
2. Escribe tu nombre y elige cuántas IAs quieres enfrentar (1–3).
3. Haz clic en **"Nueva Partida"**.

### Fases del turno

| Fase | Qué hacer |
|------|-----------|
| **Refuerzo** | Haz clic en tus territorios para añadir tropas |
| **Ataque** | Selecciona tu territorio → territorio enemigo → confirma |
| **Fortificación** | Mueve tropas entre tus territorios conectados |

El objetivo es **conquistar todos los territorios** del mapa.

---

## 🤖 La IA

Cada IA usa una **cadena de Markov de 4 estados** para decidir su estrategia:

- **EQUILIBRIO** — Consolida fronteras, ataca territorios débiles
- **EXPANSIÓN** — Ataca agresivamente hacia nuevos continentes  
- **DOMINANCIA** — Presiona al jugador más fuerte
- **CRÍTICO** — Modo defensivo, reagrupa tropas

Las probabilidades de combate se calculan analíticamente con **MarkovCombat**, que modela cada batalla como una cadena de Markov.

---

## 🗺️ El mapa

20 territorios distribuidos en 3 continentes:

| Continente | Territorios | Bonus de tropas |
|------------|-------------|-----------------|
| Norteamérica | 9 | +5 por turno |
| Sudamérica | 5 | +2 por turno |
| África | 6 | +3 por turno |

---

## 🛠️ Desarrollo local

Si quieres correrlo en tu computador:

```bash
# 1. Clonar el repo
git clone https://github.com/TU_USUARIO/risk-web.git
cd risk-web

# 2. Crear entorno virtual e instalar dependencias
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Correr el servidor
python -m flask --app frontend.flask_app run --debug

# 4. Abrir en el navegador
# http://localhost:5000
```

---

## 📁 Estructura del proyecto

```
risk-web/
├── frontend/
│   ├── flask_app.py          # API REST (Flask)
│   ├── templates/
│   │   └── index.html        # Interfaz del juego
│   └── static/
│       ├── map.js            # Lógica del mapa SVG
│       ├── mapa.png          # Imagen del mapa
│       └── territories/      # Imágenes por territorio
├── game/
│   ├── engine.py             # Motor principal del juego
│   ├── player.py             # Modelo de jugador
│   └── state.py              # Serialización de estado
├── graph/
│   └── world_graph.py        # Grafo del mapa (networkx)
├── simulation/
│   ├── combat.py             # Combate + cadenas de Markov
│   └── markov_ai.py          # Agente IA
├── data/
│   └── map_data.py           # Territorios y fronteras
├── requirements.txt
├── Procfile
└── railway.json
```
