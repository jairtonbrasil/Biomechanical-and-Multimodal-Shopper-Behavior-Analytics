"""Generate local deployment credentials while preserving existing database passwords."""

import argparse
import os
import secrets
from pathlib import Path


def generated_values() -> dict[str, str]:
    values = {
        name: secrets.token_hex(24)
        for name in (
            "DATABASE_PASSWORD",
            "REDIS_PASSWORD",
            "MEDIAMTX_API_PASSWORD",
            "MEDIAMTX_READ_PASSWORD",
            "MEDIAMTX_PUBLISH_PASSWORD",
        )
    }
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Append missing settings without rotating existing credentials",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / ".env"
    template = (root / ".env.example").read_text(encoding="utf-8")
    generated = generated_values()
    if arguments.upgrade:
        if not destination.is_file():
            raise SystemExit("Create initial configuration before using --upgrade")
        descriptor = os.open(destination, os.O_RDWR | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as file:
            existing = "".join(
                line
                for line in file.read().splitlines(keepends=True)
                if not line.startswith(("API_ACCESS_KEYS=", "API_TOKEN="))
            )
            file.seek(0)
            file.write(existing)
            file.truncate()
            names = {
                line.split("=", 1)[0]
                for line in existing.splitlines()
                if "=" in line and not line.startswith("#")
            }
            additions = []
            for line in template.splitlines():
                if "=" not in line or line.startswith("#"):
                    continue
                name, default = line.split("=", 1)
                if name not in names:
                    additions.append(f"{name}={generated.get(name, default)}")
            if additions:
                file.write(
                    ("" if existing.endswith("\n") else "\n")
                    + "\n".join(additions)
                    + "\n"
                )
            os.fchmod(file.fileno(), 0o600)
        print(
            f"Added {len(additions)} missing settings; existing credentials were preserved"
        )
        return
    for name, value in generated.items():
        template = template.replace(f"{name}=\n", f"{name}={value}\n")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(template)
    except FileExistsError:
        raise SystemExit(
            "An .env file already exists; use --upgrade to append missing settings"
        ) from None
    print("Created .env with independent local credentials; values are not displayed")


if __name__ == "__main__":
    main()
