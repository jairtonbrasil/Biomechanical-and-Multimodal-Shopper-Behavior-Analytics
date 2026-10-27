"""Prepare checksum-pinned local model artifacts without runtime downloads."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: Path, output: Path, artifact: dict, *, download: bool) -> None:
    target = output / artifact["filename"]
    if target.is_file() and checksum(target) == artifact["sha256"]:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".model-", dir=output)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            if source.is_file():
                if source.stat().st_size > MAX_ARTIFACT_BYTES:
                    raise ValueError("Source artifact exceeds its size limit")
                with source.open("rb") as local:
                    shutil.copyfileobj(local, destination)
            elif download:
                # Only reviewed manifest URLs are used; downloaded bytes must match the pinned hash.
                with urlopen(artifact["source"], timeout=30) as remote:
                    count = 0
                    while chunk := remote.read(1024 * 1024):
                        count += len(chunk)
                        if count > MAX_ARTIFACT_BYTES:
                            raise ValueError(
                                "Downloaded artifact exceeds its size limit"
                            )
                        destination.write(chunk)
            else:
                raise FileNotFoundError(
                    "Local artifact is unavailable; supply its source path"
                )
            destination.flush()
            os.fsync(destination.fileno())
        temporary_path = Path(temporary)
        if checksum(temporary_path) != artifact["sha256"]:
            raise ValueError("Model checksum differs from the reviewed manifest")
        # Public model weights must be readable by the unprivileged container UID.
        temporary_path.chmod(0o644)
        temporary_path.replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-source", type=Path, default=ROOT / "notebooks/yolo11s-pose.pt"
    )
    parser.add_argument(
        "--face-source",
        type=Path,
        default=Path.home() / ".hsemotion/enet_b0_8_best_vgaf.onnx",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow explicit download from pinned official sources",
    )
    arguments = parser.parse_args()
    manifest = json.loads((ROOT / "infrastructure/vision-models.json").read_text())
    arguments.output.mkdir(parents=True, exist_ok=True)
    try:
        for name, source in (
            ("pose", arguments.pose_source),
            ("face", arguments.face_source),
        ):
            prepare(
                source, arguments.output, manifest[name], download=arguments.download
            )
        (arguments.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"Model preparation failed: {type(error).__name__}") from None
    for name in ("pose", "face"):
        artifact = manifest[name]
        print(f"Verified {artifact['filename']} SHA256 {artifact['sha256']}")
    print("Runtime model downloads are disabled; artifacts are prepared locally")


if __name__ == "__main__":
    main()
