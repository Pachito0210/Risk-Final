"""
map_data.py
Definición estática del mapa: territorios, continentes y fronteras.
Fuente de verdad para construir el WorldGraph.
Actualizado para el nuevo mapa con 20 territorios.
"""

# Territorios agrupados por continente
# Cada territorio: (id, nombre_largo, pos_x, pos_y)
CONTINENTS = {
    "North America": {
        "bonus": 5,
        "territories": [
            ("ALA", "Alaska",                      290, 210),
            ("TNO", "Territorios del Noroeste",    380, 195),
            ("GRN", "Groenlandia",                 515, 145),
            ("ALB", "Alberta",                     360, 280),
            ("ONT", "Ontario",                     410, 295),
            ("QUE", "Quebec",                      465, 305),
            ("OEU", "Oeste de EEUU",               365, 390),
            ("EEU", "Este de EEUU",                420, 415),
            ("CAM", "Centroamérica",               370, 520),
        ],
    },
    "South America": {
        "bonus": 2,
        "territories": [
            ("VEN", "Venezuela",                   445, 615),
            ("COL", "Colombia",                    410, 625),
            ("BRA", "Brasil",                      475, 710),
            ("PER", "Perú",                        425, 730),
            ("ARG", "Argentina",                   440, 885),
        ],
    },
    "Africa": {
        "bonus": 3,
        "territories": [
            ("NAF", "Norte de África",             610, 675),
            ("EGY", "Egipto",                      665, 630),
            ("CON", "Congo",                       665, 805),
            ("EAF", "África del Este",             700, 760),
            ("SAF", "Sudáfrica",                   670, 925),
            ("MAD", "Madagascar",                  735, 930),
        ],
    },
}

# Fronteras: lista de pares (id_A, id_B)
BORDERS = [
    # ── Internas North America ──────────────────────────
    ("ALA", "TNO"),
    ("ALA", "ALB"),
    ("TNO", "ALB"),
    ("TNO", "ONT"),
    ("TNO", "GRN"),
    ("GRN", "QUE"),
    ("ALB", "ONT"),
    ("ALB", "OEU"),
    ("ONT", "QUE"),
    ("ONT", "EEU"),
    ("QUE", "EEU"),
    ("OEU", "EEU"),
    ("OEU", "CAM"),
    ("EEU", "CAM"),

    # ── Internas South America ──────────────────────────
    ("VEN", "COL"),
    ("VEN", "BRA"),
    ("COL", "BRA"),
    ("COL", "PER"),
    ("BRA", "PER"),
    ("BRA", "ARG"),
    ("PER", "ARG"),

    # ── Internas Africa ─────────────────────────────────
    ("NAF", "EGY"),
    ("NAF", "CON"),
    ("EGY", "EAF"),
    ("EGY", "CON"),
    ("CON", "EAF"),
    ("CON", "SAF"),
    ("EAF", "SAF"),
    ("EAF", "MAD"),
    ("SAF", "MAD"),

    # ── Inter-continentales ──────────────────────────────
    ("CAM", "VEN"),           # N. América → S. América
    ("BRA", "NAF"),           # S. América → África (ruta atlántica)
]
