"""Create local configuration without overwriting an existing environment file."""

import os
import secrets
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = (root / ".env.example").read_text(encoding="utf-8")
    for name in ("DATABASE_PASSWORD", "REDIS_PASSWORD"):
        configuration = configuration.replace(
            f"{name}=\n", f"{name}={secrets.token_hex(24)}\n"
        )
    try:
        descriptor = os.open(root / ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(configuration)
    except FileExistsError:
        raise SystemExit(
            "An .env file already exists; review it without overwriting credentials"
        ) from None
    print("Created .env with independent local credentials; values are not displayed")


if __name__ == "__main__":
    main()
