"""Exercise the operational API and render its persisted polygons on a real reference image."""

import argparse
import base64
import html
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def local_configuration() -> dict[str, str]:
    values = {}
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                values[name] = value.strip().strip("'")
    return {**values, **os.environ}


class DemoClient:
    def __init__(self, origin: str) -> None:
        self.origin = origin.rstrip("/")

    def call(
        self, method: str, path: str, data: dict | None = None
    ) -> tuple[int, bytes, dict]:
        request = Request(
            self.origin + path,
            method=method,
            headers={
                "Content-Type": "application/json",
            },
            data=json.dumps(data).encode() if data is not None else None,
        )
        try:
            with urlopen(request, timeout=20) as response:
                return response.status, response.read(), dict(response.headers)
        except HTTPError as error:
            return error.code, error.read(), dict(error.headers)

    def json(
        self, method: str, path: str, data: dict | None = None, expected: int = 200
    ):
        code, content, _ = self.call(method, path, data)
        if code != expected:
            raise RuntimeError(
                f"API demonstration failed: {method} {path} returned HTTP {code}"
            )
        return json.loads(content)


def render(
    output: Path, image: bytes, width: int, height: int, shelves: list[dict]
) -> None:
    encoded = base64.b64encode(image).decode()
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{encoded}"/>',
    ]
    for shelf in shelves:
        for field, color in (("product_roi", "#4bc0ff"), ("presence_roi", "#86efac")):
            points = " ".join(
                f"{x * width:.2f},{y * height:.2f}" for x, y in shelf[field]["points"]
            )
            pieces.append(
                f'<polygon points="{points}" fill="{color}" fill-opacity="0.14" stroke="{color}" stroke-width="3"/>'
            )
        x, y = shelf["product_roi"]["points"][0]
        label = html.escape(f"{shelf['name']} — ROI v{shelf['roi_version']}")
        pieces.append(
            f'<text x="{max(5, x * width):.2f}" y="{max(20, y * height):.2f}" font-family="sans-serif" font-size="15" fill="white" stroke="black" stroke-width="0.4">{label}</text>'
        )
    pieces.append("</svg>")
    (output / "frame-with-rois.svg").write_text("\n".join(pieces))
    rows = "".join(
        f"<tr><td>{html.escape(s['name'])}</td><td>{s['roi_version']}</td></tr>"
        for s in shelves
    )
    (output / "index.html").write_text(f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Stage 3 Configuration Verification</title>
<style>body{{font:16px system-ui;max-width:1280px;margin:2rem auto;padding:0 1rem;background:#111827;color:#f9fafb}}img{{width:100%;height:auto}}td,th{{padding:.5rem;text-align:left}}a{{color:#7dd3fc}}</style>
<h1>Stage 3 Configuration Verification</h1>
<p>Reference image captured through MediaMTX. Polygons retrieved from the operational API and persisted in PostgreSQL.</p>
<p>Blue: product ROI. Green: presence ROI. This page verifies configuration alignment; it does not run behavioral inference.</p>
<img src="frame-with-rois.svg" alt="Camera reference image with persisted product and presence polygons">
<table><tr><th>Shelf</th><th>ROI version</th></tr>{rows}</table>
<p><a href="verification.json">Verification record</a></p></html>""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage3")
    args = parser.parse_args()
    origin = urlsplit(args.base_url)
    if origin.scheme != "http" or origin.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise SystemExit("This demo requires a local HTTP endpoint")
    configuration = local_configuration()
    client = DemoClient(args.base_url)
    cameras = client.json("GET", "/api/v1/cameras?limit=100")
    matches = [camera for camera in cameras if camera["name"] == "Stage 3 Retail Demo"]
    if len(matches) > 1:
        raise SystemExit("Multiple demo cameras exist; review them before retrying")
    if matches:
        camera = matches[0]
        if not camera["is_active"]:
            camera = client.json(
                "PUT",
                f"/api/v1/cameras/{camera['id']}",
                {
                    "expected_revision": camera["configuration_revision"],
                    "is_active": True,
                },
            )
    else:
        password = quote(configuration["MEDIAMTX_READ_PASSWORD"], safe="")
        camera = client.json(
            "POST",
            "/api/v1/cameras",
            {
                "name": "Stage 3 Retail Demo",
                "is_active": True,
                "stream_url": f"rtsp://media-reader:{password}@127.0.0.1:8554/demo/retail",
            },
            expected=201,
        )
    camera_id = camera["id"]
    deadline = time.monotonic() + 45
    while True:
        runtime = client.json("GET", f"/api/v1/cameras/{camera_id}/runtime")
        if runtime["proxy_state"] == "APPLIED":
            code, image, headers = client.call(
                "GET", f"/api/v1/cameras/{camera_id}/frame-zero"
            )
            if code == 200:
                break
        if time.monotonic() >= deadline:
            raise SystemExit(
                "No reference image became available; inspect the demo source, reconciler and capture worker"
            )
        time.sleep(1)
    reference = next(
        value for key, value in headers.items() if key.lower() == "x-reference-revision"
    )
    width = int(
        next(value for key, value in headers.items() if key.lower() == "x-frame-width")
    )
    height = int(
        next(value for key, value in headers.items() if key.lower() == "x-frame-height")
    )
    shelves = client.json("GET", f"/api/v1/shelves?camera_id={camera_id}&limit=100")
    for specification in json.loads(
        (ROOT / "infrastructure/demo_shelves.json").read_text()
    ):
        matching = [
            shelf for shelf in shelves if shelf["name"] == specification["name"]
        ]
        if matching:
            if matching[0]["product_roi"]["reference_revision"] != reference:
                raise SystemExit(
                    "Existing demo geometry uses another reference; recalibrate it explicitly"
                )
            continue
        client.json(
            "POST",
            "/api/v1/shelves",
            {
                **specification,
                "camera_id": camera_id,
                "product_roi": {
                    **specification["product_roi"],
                    "reference_revision": reference,
                },
            },
            expected=201,
        )
    shelves = client.json("GET", f"/api/v1/shelves?camera_id={camera_id}&limit=100")
    _, repeated_image, repeated_headers = client.call(
        "GET", f"/api/v1/cameras/{camera_id}/frame-zero"
    )
    assert image == repeated_image, "Reference changed without an explicit refresh"
    assert repeated_headers["x-reference-revision"] == reference
    checks = {
        "stable_reference": True,
        "shelf_count": len(shelves),
        "camera_id": camera_id,
        "reference_revision": reference,
        "image_width": width,
        "image_height": height,
        "shelves": [
            {key: shelf[key] for key in ("id", "name", "roi_version", "is_active")}
            for shelf in shelves
        ],
    }
    checks["stale_geometry_status"] = client.call(
        "PUT",
        f"/api/v1/shelves/{shelves[0]['id']}",
        {"expected_version": shelves[0]["roi_version"] + 1, "name": shelves[0]["name"]},
    )[0]
    assert checks["stale_geometry_status"] == 409
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frame-zero.jpg").write_bytes(image)
    (args.output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    render(args.output, image, width, height, shelves)
    print(
        f"Verified {len(shelves)} persisted shelf configurations and a stable {width}x{height} reference"
    )
    print(f"Open {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
