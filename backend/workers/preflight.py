"""Check worker infrastructure without consuming or acknowledging observations."""

import argparse
import asyncio

from app.core.config import Settings
from app.core.resources import open_resources


async def check() -> None:
    async with open_resources(Settings()) as resources:
        if not await resources.ready():
            raise RuntimeError("Worker infrastructure is unavailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["vision", "temporal"])
    arguments = parser.parse_args()
    try:
        asyncio.run(check())
    except Exception as error:
        raise SystemExit(f"{arguments.role} preflight failed ({type(error).__name__})") from None
    print(
        f"{arguments.role} infrastructure preflight passed; business processing is not implemented"
    )


if __name__ == "__main__":
    main()
