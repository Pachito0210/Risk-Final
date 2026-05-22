"""
graph_visualizer.py
Visualiza el WorldGraph usando matplotlib y networkx.

Produce una imagen del mapa con:
  - Colores por jugador (gris = sin dueño)
  - Tamaño de nodo proporcional a las tropas
  - Etiquetas con id + tropas
  - Aristas internas (mismo continente) vs inter-continentales
  - Leyenda de jugadores y continentes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from graph.world_graph import WorldGraph
from data.map_data import CONTINENTS


# Paleta de colores para jugadores
PLAYER_COLORS = {
    None:      "#CCCCCC",   # sin dueño
    "player1": "#4A90D9",   # azul
    "player2": "#E05A3A",   # rojo
    "player3": "#6DBF6D",   # verde
    "player4": "#F5A623",   # naranja
    "player5": "#9B59B6",   # morado
}

# Color de fondo por continente (para el bounding box)
CONTINENT_COLORS = {
    "North America": "#D6EAF8",
    "South America": "#D5F5E3",
    "Africa":        "#FDEBD0",
}


def _node_color(owner: str | None) -> str:
    return PLAYER_COLORS.get(owner, "#AAAAAA")


def _edge_is_intercontinental(world: WorldGraph, a: str, b: str) -> bool:
    return world.continent(a) != world.continent(b)


def draw_map(
    world: WorldGraph,
    title: str = "Mapa Risk — Estado actual",
    output_path: str | None = None,
    show: bool = False,
) -> str:
    """
    Dibuja el WorldGraph y lo guarda como imagen PNG.

    Args:
        world       : instancia de WorldGraph
        title       : título de la figura
        output_path : ruta de salida (default: /tmp/risk_map.png)
        show        : si True, abre la ventana de matplotlib

    Returns:
        Ruta del archivo generado.
    """
    if output_path is None:
        output_path = "/tmp/risk_map.png"

    G = world.G

    # ── Posiciones (invertir Y porque matplotlib y ≠ pantalla) ──────────────
    pos = {
        n: (data["pos"][0], -data["pos"][1])
        for n, data in G.nodes(data=True)
    }

    # ── Atributos visuales ───────────────────────────────────────────────────
    node_colors = [_node_color(G.nodes[n]["owner"]) for n in G.nodes]
    node_sizes  = [300 + G.nodes[n]["troops"] * 120 for n in G.nodes]

    # Separar aristas internas vs inter-continentales
    internal_edges = []
    external_edges = []
    for a, b in G.edges:
        if _edge_is_intercontinental(world, a, b):
            external_edges.append((a, b))
        else:
            internal_edges.append((a, b))

    # ── Etiquetas: id + tropas ───────────────────────────────────────────────
    labels = {
        n: f"{n}\n({G.nodes[n]['troops']}T)"
        for n in G.nodes
    }

    # ── Figura ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#EBF5FB")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.axis("off")

    # Sombreado de continentes (bounding box)
    for continent, color in CONTINENT_COLORS.items():
        nodes_in = [
            n for n in G.nodes
            if G.nodes[n]["continent"] == continent
        ]
        if not nodes_in:
            continue
        xs = [pos[n][0] for n in nodes_in]
        ys = [pos[n][1] for n in nodes_in]
        margin = 22
        rect = plt.Rectangle(
            (min(xs) - margin, min(ys) - margin),
            max(xs) - min(xs) + margin * 2,
            max(ys) - min(ys) + margin * 2,
            linewidth=1.5,
            edgecolor="#AAAAAA",
            facecolor=color,
            alpha=0.4,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            min(xs) - margin + 4,
            max(ys) + margin - 6,
            continent,
            fontsize=9,
            color="#555555",
            fontstyle="italic",
        )

    # Dibujar aristas internas
    nx.draw_networkx_edges(
        G, pos, edgelist=internal_edges, ax=ax,
        edge_color="#666666", width=1.4, alpha=0.7,
    )

    # Dibujar aristas inter-continentales (punteadas)
    nx.draw_networkx_edges(
        G, pos, edgelist=external_edges, ax=ax,
        edge_color="#CC6600", width=1.6, alpha=0.8,
        style="dashed",
    )

    # Dibujar nodos
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.5,
        edgecolors="#333333",
    )

    # Etiquetas
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=7, font_weight="bold", font_color="#111111",
    )

    # ── Leyenda de jugadores ─────────────────────────────────────────────────
    owners_present = sorted(
        {G.nodes[n]["owner"] for n in G.nodes},
        key=lambda x: (x is None, x),
    )
    legend_patches = []
    for owner in owners_present:
        label = "Sin dueño" if owner is None else owner
        color = _node_color(owner)
        legend_patches.append(mpatches.Patch(color=color, label=label))

    # Leyenda de aristas
    legend_patches.append(
        mpatches.Patch(color="#666666", label="Frontera interna")
    )
    legend_patches.append(
        mpatches.Patch(color="#CC6600", label="Frontera intercontinental", linestyle="--")
    )

    ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=8,
        framealpha=0.85,
        edgecolor="#CCCCCC",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path
