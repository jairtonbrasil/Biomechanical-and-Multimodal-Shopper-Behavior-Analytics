# Database Migrations

Alembic manages the eleven domain tables and the `alembic_version` metadata table.
The Compose `migrate` job applies the current head before the API and verification
services start. SQLAlchemy's `create_all` is not used for application deployment.

From the repository root:

```bash
docker compose run --build --rm migrate
docker compose run --rm migrate alembic current
docker compose run --rm migrate alembic check
```

Revision `0001_data_layer` freezes the initial DDL independently of application
model imports. It includes restrictive foreign keys, temporal/confidence/count
checks, B-tree and GIN indexes, and triggers for immutable model versions,
append-only geometry history, camera consistency and business model kinds.

Revision `0002_english_names` standardizes table, column, constraint, index and
structured JSONB names in English. It preserves existing records, UUIDs and ROI
history, updates stored observations to contract 2.0, and reinstalls the guards.
Legacy identifiers appear only in the frozen initial revision and reversible
compatibility mappings. Tests cover populated upgrade/downgrade and atomic rollback
when JSON key conversion detects a collision.

Generate future revisions from `backend/` with `alembic revision --autogenerate`,
then review the generated DDL and any non-reflected functions/triggers manually.
Every schema change must pass isolated upgrade/downgrade and metadata-drift tests.
Downgrading to `base` removes domain tables and must not be run against an application
database as a routine verification step. See [infrastructure operations](../../docs/infrastructure.md).
