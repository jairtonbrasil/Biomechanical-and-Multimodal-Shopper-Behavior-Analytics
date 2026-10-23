import json

import pytest
from sqlalchemy import text

from app.core.database import Database
from tests.factories import example_envelope


def contract_documents() -> tuple[dict, dict]:
    full = example_envelope().model_dump(mode="json")
    baseline = json.loads(json.dumps(full))
    attributes = baseline["observation"]["attributes"]
    for group in ("provenance", "geometry", "presence", "kinematics", "spatial", "evidence"):
        del attributes[group]
    return baseline, full


def test_snapshot_payload_stays_within_message_budget() -> None:
    baseline, full = contract_documents()

    def compact(value: dict) -> bytes:
        return json.dumps(value, separators=(",", ":")).encode()

    minimum_size, full_size = len(compact(baseline)), len(compact(full))
    print(
        {
            "minimum_wire_bytes": minimum_size,
            "full_wire_bytes": full_size,
            "extension_wire_bytes": full_size - minimum_size,
        }
    )
    assert minimum_size < full_size <= 262144 - 512


@pytest.mark.integration
async def test_snapshot_jsonb_datum_overhead(domain_database: Database) -> None:
    baseline, full = contract_documents()
    async with domain_database.transaction() as session:
        sizes = []
        for value in (baseline, full):
            sizes.append(
                await session.scalar(
                    text("SELECT pg_column_size(CAST(:document AS jsonb))"),
                    {"document": json.dumps(value)},
                )
            )
    print(
        {
            "minimum_jsonb_datum_bytes": sizes[0],
            "full_jsonb_datum_bytes": sizes[1],
            "extension_jsonb_datum_bytes": sizes[1] - sizes[0],
        }
    )
    assert sizes[1] > sizes[0]
