"""Run real RTSP inference, inspect Kafka observations and verify Stage 4 artifacts."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def execute(command: list[str], **kwargs) -> None:
    subprocess.run(command, cwd=ROOT, check=True, **kwargs)


def get(path: str, *, binary: bool = False):
    with urlopen(f"http://127.0.0.1:8000{path}", timeout=12) as response:
        body = response.read()
    return body if binary else json.loads(body)


def wait_event(
    path: Path, name: str, process: subprocess.Popen, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    if json.loads(line).get("event") == name:
                        return
                except ValueError:
                    continue
        if process.poll() is not None:
            raise RuntimeError(f"Diagnostic inspector exited; inspect {path.name}")
        time.sleep(0.2)
    raise RuntimeError("Diagnostic inspector did not become ready within its deadline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 10000 or not 5 <= args.max_seconds <= 600:
        raise SystemExit(
            "Choose 1–10000 frames and 5–600 seconds for this bounded demo"
        )
    if not (ROOT / "models/manifest.json").is_file():
        raise SystemExit("Run python3 scripts/prepare_vision_models.py first")
    compose = ["docker", "compose", "--profile", "vision", "--profile", "demo"]
    if args.build:
        execute(
            compose
            + [
                "build",
                "api",
                "reference-capture",
                "demo-source",
                "vision-worker",
                "observation-inspector",
            ]
        )
    execute(compose + ["up", "-d", "--wait", "api", "demo-source"])
    execute([sys.executable, "scripts/demo_stage3.py"])
    camera = json.loads((ROOT / "outputs/stage3/verification.json").read_text())[
        "camera_id"
    ]
    directory = ROOT / "outputs/stage4"
    directory.mkdir(parents=True, exist_ok=True)
    # Grant the image's fixed non-root UID access without making output world-writable.
    if sys.platform.startswith("linux"):
        execute(["setfacl", "-m", "u:10001:rwx", str(directory)])
    tag = uuid4().hex[:8]
    observed = directory / f"inspected-{tag}.jsonl"
    consumer_log = directory / f"consumer-{tag}.jsonl"
    producer_log = directory / f"producer-{tag}.jsonl"
    consumer_name = f"shopper-inspector-{tag}"
    producer_name = f"shopper-vision-{tag}"
    with (
        consumer_log.open("w") as consumer_output,
        producer_log.open("w") as producer_output,
    ):
        inspector = subprocess.Popen(
            compose
            + [
                "run",
                "--rm",
                "--no-deps",
                "--name",
                consumer_name,
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "observation-inspector",
                "python",
                "-m",
                "workers.vision_streamer.inspect",
                "--camera-id",
                camera,
                "--limit",
                "10",
                "--timeout",
                str(int(args.max_seconds + 90)),
                "--output",
                f"/outputs/stage4/{observed.name}",
            ],
            cwd=ROOT,
            stdout=consumer_output,
            stderr=subprocess.STDOUT,
        )
        worker = None
        try:
            wait_event(consumer_log, "inspection_started", inspector, 40)
            worker = subprocess.Popen(
                compose
                + [
                    "run",
                    "--rm",
                    "--no-deps",
                    "--name",
                    producer_name,
                    "vision-worker",
                    "python",
                    "-m",
                    "workers.vision_streamer.main",
                    "--camera-id",
                    camera,
                    "--max-frames",
                    str(args.max_frames),
                    "--max-seconds",
                    str(args.max_seconds),
                ],
                cwd=ROOT,
                stdout=producer_output,
                stderr=subprocess.STDOUT,
            )
            descriptor = get(f"/api/v1/cameras/{camera}/playback")
            deadline = time.monotonic() + 30
            while True:
                try:
                    playlist = get(descriptor["hls_url"], binary=True)
                    if playlist.startswith(b"#EXTM3U"):
                        break
                except (HTTPError, URLError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError("HLS source playback did not become available")
                time.sleep(0.5)
            code = worker.wait(timeout=args.max_seconds + 120)
            if code:
                raise RuntimeError(f"Vision worker failed; inspect {producer_log.name}")
            if inspector.wait(timeout=30):
                raise RuntimeError(
                    f"Kafka inspection failed; inspect {consumer_log.name}"
                )
        finally:
            for process, name in ((worker, producer_name), (inspector, consumer_name)):
                if process is not None and process.poll() is None:
                    subprocess.run(
                        ["docker", "stop", "--time", "10", name],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    process.wait(timeout=20)
    latest = json.loads((directory / "latest.json").read_text())
    report = latest["report"]
    if (
        report["status"] != "completed"
        or report["counters"].get("observations_published", 0) <= 0
    ):
        raise RuntimeError(
            "The real footage did not produce successfully published observations"
        )
    if not observed.is_file() or not observed.read_text().strip():
        raise RuntimeError("Kafka inspection produced no observation evidence")
    run_directory = directory / latest["run_directory"]
    if not (run_directory / "annotated.mp4").is_file():
        raise RuntimeError("The worker did not produce an annotated diagnostic clip")
    verification = {
        "camera_id": camera,
        "run_directory": latest["run_directory"],
        "producer_log": producer_log.name,
        "consumer_log": consumer_log.name,
        "inspected_observations": observed.name,
        "playback_url": "http://127.0.0.1:8000" + descriptor["player_url"],
        "hls_playlist_verified": True,
        "published_observations": report["counters"]["observations_published"],
        "consumer_role": "read_only_diagnostic_not_temporal_processor",
    }
    (directory / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(
        f"Published {verification['published_observations']} observations and verified Kafka delivery"
    )
    print(f"Annotated video: {run_directory / 'annotated.mp4'}")
    print(f"Execution report: {run_directory / 'report.json'}")
    print(f"Live source: {verification['playback_url']}")
    print(f"Producer log: {producer_log}\nConsumer log: {consumer_log}")


if __name__ == "__main__":
    try:
        main()
    except (
        subprocess.CalledProcessError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        raise SystemExit(str(error)) from None
