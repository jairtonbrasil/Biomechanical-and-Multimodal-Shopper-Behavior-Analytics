"""Generate a synthetic local video fixture without downloading footage or models."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def create_fixture(output: Path, *, force: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError("Fixture exists; use --force to replace it")
    if output.is_symlink():
        raise ValueError("Fixture output must not be a symbolic link")
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if force else "-n",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=10",
        "-t",
        "15",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "10",
        "-threads",
        "2",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError(
            "Synthetic fixture generation failed; verify local FFmpeg"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/stage4-fixture.mp4"
    )
    parser.add_argument("--force", action="store_true")
    options = parser.parse_args()
    try:
        create_fixture(options.output, force=options.force)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
    print(
        f"Created synthetic 1280x720, 10 FPS, 15-second H.264 fixture: {options.output}"
    )
    print(
        "Use DEMO_VIDEO_FILE with the Compose demo profile; this fixture contains no shoppers."
    )


if __name__ == "__main__":
    main()
