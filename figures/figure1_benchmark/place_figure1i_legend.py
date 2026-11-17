"""Reconstruct the assembly SVG by applying the recorded Batik legend coordinates."""
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_HASH = "4d70d1bc5d539186ea901467d19700f750b757e2022e2276ed0c6c90f7b7b2cd"
ASSEMBLY_HASH = "9e9c12879b2dc980d0a5416569b36c5ed26b8ddd71ae919c389f15968385c51e"

def main():
    source = (ROOT / "probe_area3.svg").read_bytes()
    if sha256(source).hexdigest() != SOURCE_HASH:
        raise ValueError("Run replot_probe_area3 in MATLAB R2022a to produce the recorded Batik export.")
    svg = source.decode("utf-8")
    positions = {
        "M496 357 L496 222 L358 222 L358 357 Z": "M511 143 L511 8 L373 8 L373 143 Z",
        "M358 357 L358 222 L496 222 L496 357 Z": "M373 143 L373 8 L511 8 L511 143 Z",
        "translate(-367.2885,-347.0369)": "translate(-382.2885,-133.0369)",
    }
    # Batik prints four decimal places. Use the final assembly's stored rounded values.
    y_positions = [
        ("231.9631","17.9631"), ("248.4022","34.4022"), ("264.8413","50.8413"),
        ("281.2805","67.2804"), ("297.7195","83.7196"), ("314.1587","100.1587"),
        ("330.5978","116.5978"), ("347.0369","133.0369"),
    ]
    for old_x,new_x in [("375.5817","390.5817"),("367.2885","382.2885")]:
        for old_y,new_y in y_positions:
            positions[f"translate({old_x},{old_y})"] = f"translate({new_x},{new_y})"
    for old,new in positions.items():
        if svg.count(old) != 1:
            raise ValueError(f"Unexpected SVG legend structure at {old}")
        svg = svg.replace(old,new)
    rendered = svg.encode("utf-8")
    if sha256(rendered).hexdigest() != ASSEMBLY_HASH:
        raise ValueError("Legend-adjusted SVG did not match the preserved final asset.")
    (ROOT / "probe_area6.svg").write_bytes(rendered)

if __name__ == "__main__":
    main()
