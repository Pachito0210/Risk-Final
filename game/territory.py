@dataclass
class Territory:
    id: str              # "BRA", "USA"...
    name: str
    continent: str
    owner: str | None    # player_id
    troops: int
    position: tuple      # (x, y) para render