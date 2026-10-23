"""Validate notebook publication artifacts without running model inference."""

import ast
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "shopper_behavior.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                raise ValueError(f"Notebook cell {index} contains a saved error output")
            html = "".join(output.get("data", {}).get("text/html", []))
            if "data:video/" in html:
                raise ValueError(f"Notebook cell {index} contains an embedded video")
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Notebook exceeds the 5 MiB publication budget")
    media = root / "docs" / "media"
    for name in (
        "retail-store-preview.gif",
        "retail-store-processed.mp4",
        "aisle-closeup-preview.gif",
        "aisle-closeup-processed.mp4",
        "retail-store-heatmaps.png",
        "retail-store-kinematics.png",
        "aisle-closeup-heatmaps.png",
        "aisle-closeup-kinematics.png",
    ):
        if not (media / name).is_file() or (media / name).stat().st_size == 0:
            raise ValueError(f"Missing demonstration artifact: {name}")
    print("Notebook syntax, saved outputs, size, and demonstration artifacts verified")


if __name__ == "__main__":
    main()
