import ast
import json
from pathlib import Path

import pytest

from workers.demo_seed import shelf_specifications

ROOT = Path(__file__).resolve().parents[2]


def test_iprox_preset_preserves_notebook_geometry_names_and_categories():
    notebook = json.loads((ROOT / "notebooks/shopper_behavior.ipynb").read_text())
    original = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SHELVES_CONFIG_IPROX"
                for target in node.targets
            ):
                original = ast.literal_eval(node.value)
    assert original is not None
    accepted = shelf_specifications(ROOT / "infrastructure/demo_shelves.json")
    assert len(accepted) == len(original) == 5
    for actual, expected in zip(accepted, original, strict=True):
        assert actual["name"] == expected["name"]
        assert actual["category"] == expected["category"]
        assert actual["product_roi"] == expected["product_roi"]
        assert actual["presence_roi"] == {
            key: value
            for key, value in expected["presence_roi"].items()
            if key != "reference_point"
        }


def test_demo_source_and_fixture_are_selected_together():
    scenario = json.loads((ROOT / "infrastructure/demo_scenario.json").read_text())
    assert scenario["camera_name"] == "IPROX_HD_RETAIL_AISLE_01"
    assert scenario["video_file"] == "notebooks/iprox_retail_aisle_hd.mp4"
    assert scenario["stream_path"] == "demo/iprox"
    assert len(shelf_specifications(ROOT / scenario["shelves_file"])) == 5


def test_invalid_seed_geometry_is_rejected_before_database_writes(tmp_path):
    data = json.loads((ROOT / "infrastructure/demo_shelves.json").read_text())
    data[0]["product_roi"]["points"] = [[0, 0], [0.5, 0.5], [1, 1]]
    fixture = tmp_path / "shelves.json"
    fixture.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        shelf_specifications(fixture)
