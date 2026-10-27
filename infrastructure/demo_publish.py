"""Publish a local, unannotated video as an authenticated demonstration camera."""

import os
import subprocess
import time


def main() -> None:
    source = os.environ.get("DEMO_VIDEO_PATH", "/demo/input.mp4")
    password = os.environ["MEDIAMTX_PUBLISH_PASSWORD"]
    destination = f"rtsp://demo-publisher:{password}@mediamtx:8554/demo/retail"
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-re",
        "-stream_loop",
        "-1",
        "-i",
        source,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        "30",
        "-threads",
        "2",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        destination,
    ]
    # stderr may contain credential-bearing URLs from FFmpeg; report only an exit code.
    while True:
        result = subprocess.run(
            command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(
            f"Demo publisher exited with code {result.returncode}; retrying", flush=True
        )
        time.sleep(2)


if __name__ == "__main__":
    main()
