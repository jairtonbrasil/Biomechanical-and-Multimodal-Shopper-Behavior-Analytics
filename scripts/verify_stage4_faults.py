"""Verify local Stage 4 reconnects, geometry cutover and bounded broker failure."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "--profile", "vision", "--profile", "demo"]


class FaultVerification:
    def __init__(self, camera_id: UUID, directory: Path) -> None:
        self.camera_id = str(camera_id)
        self.directory = directory
        self.deadline = time.monotonic() + 180
        self.workers = []
        self.publisher_stopped = False
        self.kafka_stopped = False
        self.phase = "initialization"
        self.report = {"camera_id": self.camera_id, "checks": {}, "status": "running"}

    def remaining(self, maximum: float = 30) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "Fault verification reached its three-minute execution deadline"
            )
        return min(maximum, remaining)

    @staticmethod
    def command(arguments: list[str], *, deadline: float = 25) -> None:
        subprocess.run(
            arguments,
            cwd=ROOT,
            check=True,
            timeout=deadline,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def api(path: str, method: str = "GET", data: dict | None = None):
        request = Request(
            "http://127.0.0.1:8000" + path,
            method=method,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data).encode() if data is not None else None,
        )
        with urlopen(request, timeout=8) as response:
            return json.load(response)

    @staticmethod
    def lines(path: Path) -> list[dict]:
        if not path.exists():
            return []
        if path.stat().st_size > 64 * 1024 * 1024:
            raise RuntimeError("A diagnostic artifact exceeded its review budget")
        result = []
        for line in path.read_text().splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                result.append(value)
        return result

    def wait(self, predicate, description: str, *, maximum: float = 30, process=None):
        deadline = time.monotonic() + self.remaining(maximum)
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            if process is not None and process.poll() is not None:
                raise RuntimeError(f"Worker exited before {description}")
            time.sleep(0.15)
        raise RuntimeError(f"Timed out waiting for {description}")

    def event(self, path: Path, event: str, process, *, predicate=None, maximum=30):
        def find():
            return next(
                (
                    line
                    for line in self.lines(path)
                    if line.get("event") == event
                    and (predicate is None or predicate(line))
                ),
                None,
            )

        return self.wait(find, event, maximum=maximum, process=process)

    def start_worker(self, name: str, seconds: int):
        output = self.directory / name
        output.mkdir()
        if sys.platform.startswith("linux"):
            self.command(["setfacl", "-m", "u:10001:rwx", str(output)])
        log_path = self.directory / f"{name}.jsonl"
        log = log_path.open("w")
        container = f"shopper-stage4-fault-{name}-{uuid4().hex[:8]}"
        relative = output.relative_to(ROOT / "outputs")
        environment = [
            "VISION_PREVIEW_ENABLED=false",
            "VISION_OBSERVATION_SAMPLE_LIMIT=1000",
            "VISION_SUMMARY_SECONDS=1",
            "VISION_CAPTURE_TIMEOUT_SECONDS=5",
            "VISION_RECONNECT_MAX_SECONDS=2",
            "VISION_RECONNECT_ATTEMPTS=10",
        ]
        if name == "broker":
            environment += [
                "VISION_PUBLISH_TIMEOUT_SECONDS=1",
                "VISION_PUBLISH_ATTEMPTS=2",
                "KAFKA_REQUEST_TIMEOUT_MS=3000",
            ]
        command = COMPOSE + ["run", "--rm", "--no-deps", "--name", container]
        for item in environment:
            command += ["-e", item]
        command += [
            "vision-worker",
            "python",
            "-m",
            "workers.vision_streamer.main",
            "--camera-id",
            self.camera_id,
            "--max-frames",
            "0",
            "--max-seconds",
            str(seconds),
            "--output",
            f"/outputs/{relative.as_posix()}",
        ]
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        self.workers.append((process, container, log))
        return process, output, log_path

    def run_artifacts(self, output: Path, log: Path):
        started = next(
            line for line in self.lines(log) if line.get("event") == "vision_starting"
        )
        identifier = started["run_id"]
        if Path(identifier).name != identifier:
            raise RuntimeError("Worker report directory is invalid")
        run = output / identifier
        return json.loads((run / "report.json").read_text()), self.lines(
            run / "observations.jsonl"
        )

    def validate_demo(self):
        camera = self.api(f"/api/v1/cameras/{self.camera_id}")
        scenario = json.loads((ROOT / "infrastructure/demo_scenario.json").read_text())
        if camera["name"] != scenario["camera_name"] or not camera["is_active"]:
            raise RuntimeError(
                "Fault checks are restricted to the active local retail demo camera"
            )
        shelves = self.api(f"/api/v1/shelves?camera_id={self.camera_id}")
        expected = json.loads((ROOT / "infrastructure/demo_shelves.json").read_text())
        if len(shelves) != 5 or {item["name"] for item in shelves} != {
            item["name"] for item in expected
        }:
            raise RuntimeError(
                "Fault checks require exactly the five known demo shelves"
            )
        if not all(item["is_active"] for item in shelves):
            raise RuntimeError(
                "Activate all five aligned demo shelves before fault verification"
            )
        self.api("/health/ready")
        return shelves

    def reconnect_and_cutover(self, shelves):
        self.phase = "reconnect_and_geometry_cutover"
        print(
            "Checking source reconnect, tracking epochs and geometry cutover.",
            flush=True,
        )
        process, output, log = self.start_worker("reconnect", 55)
        self.event(log, "producer_acknowledged", process)
        first_epoch = self.event(log, "vision_tracking_epoch", process)
        self.publisher_stopped = True
        self.command(COMPOSE + ["stop", "--timeout", "1", "demo-source"])
        time.sleep(3)
        self.command(COMPOSE + ["start", "demo-source"])
        self.publisher_stopped = False
        second_epoch = self.event(
            log,
            "vision_tracking_epoch",
            process,
            predicate=lambda line: line["source_epoch"] != first_epoch["source_epoch"],
        )
        if first_epoch["tracking_session_id"] == second_epoch["tracking_session_id"]:
            raise RuntimeError(
                "A source reconnect retained its previous tracking session"
            )
        shelf = shelves[0]
        product = {
            key: shelf["product_roi"][key]
            for key in ("type", "coordinate_system", "points", "reference_revision")
        }
        changed = self.api(
            f"/api/v1/shelves/{shelf['id']}",
            "PUT",
            {
                "expected_version": shelf["roi_version"],
                "product_roi": product,
                "presence_roi": shelf["presence_roi"],
            },
        )
        if changed["roi_version"] != shelf["roi_version"] + 1:
            raise RuntimeError(
                "The geometry edit did not advance its optimistic version"
            )
        if changed["product_roi"]["points"] != product["points"]:
            raise RuntimeError(
                "The fault check unexpectedly changed demo polygon coordinates"
            )
        if changed["product_roi"]["history"][-1]["roi_version"] != shelf["roi_version"]:
            raise RuntimeError(
                "The geometry cutover did not preserve its previous snapshot"
            )
        self.event(log, "vision_configuration_cutover", process)
        code = process.wait(timeout=self.remaining(65))
        if code:
            raise RuntimeError(
                "Reconnect/cutover worker failed; inspect reconnect.jsonl"
            )
        report, samples = self.run_artifacts(output, log)
        if report["status"] != "completed" or report["capture"]["reconnects"] < 1:
            raise RuntimeError(
                "The worker did not complete a measured source reconnect"
            )
        self.validate_geometry_samples(samples, shelves, shelf, changed)
        self.report["checks"]["source_reconnect"] = {
            "passed": True,
            "source_epochs_changed": True,
            "tracking_sessions_changed": True,
            "reconnects": report["capture"]["reconnects"],
        }
        self.report["checks"]["geometry_cutover"] = {
            "passed": True,
            "shelf_id": shelf["id"],
            "old_roi_version": shelf["roi_version"],
            "new_roi_version": changed["roi_version"],
            "coordinates_preserved": True,
            "old_snapshot_preserved": True,
            "mixed_frame_geometries": False,
        }

    @staticmethod
    def validate_geometry_samples(samples, shelves, previous, changed):
        expected_ids = {item["id"] for item in shelves}
        versions, frames = set(), {}
        boundary = datetime.fromisoformat(
            changed["product_roi"]["history"][-1]["valid_until"]
        )
        for envelope in samples:
            item = envelope["observation"]
            geometry = item["attributes"]["geometry"]
            if (
                len(geometry) != 5
                or {entry["shelf_id"] for entry in geometry} != expected_ids
            ):
                raise RuntimeError(
                    "An observation omitted or mixed the five demo geometry snapshots"
                )
            snapshot = next(
                entry for entry in geometry if entry["shelf_id"] == previous["id"]
            )
            version = snapshot["roi_version"]
            versions.add(version)
            observed_at = datetime.fromisoformat(item["observed_at"])
            if version == previous["roi_version"] and observed_at >= boundary:
                raise RuntimeError(
                    "Old geometry was published for a frame after its cutover"
                )
            if version == changed["roi_version"] and observed_at < boundary:
                raise RuntimeError(
                    "New geometry was published for a frame before its cutover"
                )
            key = (item["tracking_session_id"], item["frame_seq"])
            fingerprint = tuple(
                sorted(
                    (
                        entry["shelf_id"],
                        entry["roi_version"],
                        entry["reference_revision"],
                    )
                    for entry in geometry
                )
            )
            if key in frames and frames[key] != fingerprint:
                raise RuntimeError(
                    "One processed frame used incompatible geometry snapshots"
                )
            frames[key] = fingerprint
        if versions != {previous["roi_version"], changed["roi_version"]}:
            raise RuntimeError(
                "The bounded observation sample does not cover both geometry versions"
            )

    def broker_failure(self):
        self.phase = "broker_failure"
        print("Checking bounded broker retries and safe worker failure.", flush=True)
        process, output, log = self.start_worker("broker", 35)
        self.event(log, "producer_acknowledged", process)
        self.kafka_stopped = True
        self.command(COMPOSE + ["stop", "--timeout", "1", "kafka"])
        self.event(log, "producer_retry", process, maximum=15)
        code = process.wait(timeout=self.remaining(25))
        report, _ = self.run_artifacts(output, log)
        lines = self.lines(log)
        retries = [line for line in lines if line.get("event") == "producer_retry"]
        first_retry = next(
            index
            for index, line in enumerate(lines)
            if line.get("event") == "producer_retry"
        )
        if (
            not code
            or report["status"] != "failed"
            or report["failure_type"] != "PublicationUnavailable"
        ):
            raise RuntimeError(
                "Broker failure did not produce the expected bounded publication failure"
            )
        if [line["attempt"] for line in retries] != [1, 2]:
            raise RuntimeError(
                "Broker publication retries exceeded or missed their configured bound"
            )
        if any(
            line.get("event") == "producer_acknowledged" for line in lines[first_retry:]
        ):
            raise RuntimeError(
                "The worker reported a publication acknowledgement after broker failure"
            )
        self.command(COMPOSE + ["start", "kafka"])
        self.kafka_stopped = False
        self.wait_ready()
        self.report["checks"]["broker_failure"] = {
            "passed": True,
            "worker_exit_code": code,
            "failure_type": report["failure_type"],
            "publication_attempts": len(retries),
            "acknowledgements_after_failure": 0,
            "api_ready_after_recovery": True,
        }

    def wait_ready(self):
        def ready():
            try:
                return self.api("/health/ready").get("status") == "ready"
            except (HTTPError, URLError, TimeoutError):
                return False

        return self.wait(ready, "API readiness after service recovery", maximum=40)

    def cleanup(self):
        for process, container, log in self.workers:
            if process.poll() is None:
                try:
                    self.command(
                        ["docker", "stop", "--time", "5", container], deadline=12
                    )
                except (subprocess.SubprocessError, OSError):
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        check=False,
                        timeout=10,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            log.close()
        for stopped, service in (
            (self.publisher_stopped, "demo-source"),
            (self.kafka_stopped, "kafka"),
        ):
            if stopped:
                self.command(COMPOSE + ["start", service], deadline=20)
        self.publisher_stopped = self.kafka_stopped = False

    def execute(self):
        started = time.monotonic()
        try:
            shelves = self.validate_demo()
            self.reconnect_and_cutover(shelves)
            self.broker_failure()
            self.report["status"] = "passed"
        except (
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            StopIteration,
            OSError,
            subprocess.SubprocessError,
            KeyboardInterrupt,
        ) as error:
            self.report.update(
                {
                    "status": "failed",
                    "phase": self.phase,
                    "error_type": type(error).__name__,
                    "detail": str(error)
                    if isinstance(error, RuntimeError)
                    else "Fault check failed; inspect the sanitized case logs",
                }
            )
        finally:
            try:
                self.cleanup()
            except (OSError, subprocess.SubprocessError) as error:
                self.report.update(
                    {"status": "failed", "cleanup_error_type": type(error).__name__}
                )
            self.report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            (self.directory / "report.json").write_text(
                json.dumps(self.report, indent=2) + "\n"
            )
        print(
            f"Fault verification {self.report['status']}; report: {self.directory / 'report.json'}"
        )
        return self.report["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", type=UUID)
    options = parser.parse_args()
    camera_id = options.camera_id
    if camera_id is None:
        camera_id = UUID(
            json.loads((ROOT / "outputs/stage3/verification.json").read_text())[
                "camera_id"
            ]
        )
    directory = ROOT / "outputs/stage4/faults" / uuid4().hex[:8]
    directory.mkdir(parents=True)
    if not FaultVerification(camera_id, directory).execute():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
