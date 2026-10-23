# Local Infrastructure Operations

Run commands from the repository root. The Compose configuration uses PostgreSQL 16,
Redis, Kafka KRaft and MediaMTX/FFmpeg, plus a shared backend image for the API,
migrations and worker infrastructure preflight. The frontend workspace is reserved;
no dashboard container is currently provided.

## Configuration

Create `.env` with `python3 scripts/configure_local.py` on first setup. It contains
independent generated credentials and has owner-only permissions. Existing files
are never overwritten. All application settings load through Pydantic Settings.
Never commit `.env`, model artifacts, footage, backups or generated analytics.

| Setting | Default / constraint |
|---|---|
| API port | `API_PORT=8000`, host loopback only |
| PostgreSQL | `postgres:5432`; database/user configurable; required password |
| Database pool | 5 connections + 5 overflow; bounded connection/query waits |
| Redis | `redis:6379`; required password; 20 application connections |
| Redis memory | 256 MiB; no eviction; no persistence |
| Cache TTL | 120 seconds, validated range 60–300 |
| State TTL | 60 seconds; must exceed the track-loss tolerance |
| Kafka | `kafka:9092`; 3 observation partitions; single-node replication factor 1 |
| Kafka retention | 24 hours locally; time-based deletion is segment-based |
| Observation budget | 256 KiB default including transport allowance |
| Debounce configuration | Entry 500 ms, exit 2000 ms, track loss 5000 ms |
| Sampling configuration | 10 FPS; inference is not active yet |

Kafka plaintext transport is restricted to the local Docker bridge. No internal
service ports are published. MediaMTX has no camera paths or media permissions;
its management health endpoint is container-local. This configuration does not
implement production identity, playback authorization, TLS, high availability or
hardware sizing. Redis state and cache share capacity; a full instance rejects
writes rather than evicting temporal state.

## Startup and Migrations

```bash
docker compose config --quiet
docker compose up -d --build --wait api
docker compose ps -a
docker compose logs --tail 50 migrate api
```

The dependency order is PostgreSQL healthy → one-shot `migrate` success → API and
verification services. Redis, Kafka topic initialization and MediaMTX also have
readiness dependencies. The migration job takes a PostgreSQL advisory transaction
lock so simultaneous migration jobs serialize. Schema creation does not run inside
API request handlers or through `metadata.create_all` at startup.

```bash
# Inspect the applied revision and compare it with ORM metadata.
docker compose run --rm migrate alembic current
docker compose run --rm migrate alembic check

# Apply pending migrations explicitly when operating only the data layer.
docker compose up -d postgres
docker compose run --build --rm migrate

# Inspect tables and observation indexes without publishing PostgreSQL.
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt"'
docker compose exec postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\d frame_observation"'
```

The current head is `0002_english_names`. The migration chain defines eleven domain
tables plus Alembic metadata, UUID/composite keys, restrictive foreign keys, B-tree/GIN indexes, check
constraints and integrity triggers. Deep JSONB validation belongs to Pydantic and
the ORM write boundary; SQL checks enforce required groups and relational rules.
Manual SQL writes must not bypass the documented nested contracts.

Revision `0001_data_layer` is retained unchanged as migration history. Revision
`0002_english_names` renames domain tables, columns, indexes and constraints in
place, converts historical shelf references in JSONB, and reinstalls the integrity
guards. Existing UUIDs and relationships are preserved. Its downgrade restores
the prior naming and observation contract; downgrading all the way to `base` drops
the domain tables.

This is a coordinated application/schema change. Back up an existing database,
stop the API and any writers, and deploy the matching application image before
resuming traffic:

```bash
docker compose stop api
docker compose build api migrate
docker compose run --rm migrate
docker compose up -d --wait api
curl --fail http://127.0.0.1:8000/health/ready
```

Observation producers must emit contract 2.0 with English keys before processing
resumes. The current worker services are preflight commands rather than producers.

The API checks the required revision as part of `/health/ready`; a missing or
incompatible schema is unavailable. Updating the required application revision
belongs to future schema changes. `/health/live` reports process liveness and does
not depend on service availability.

## Verification

```bash
docker compose --profile verification run --build --rm tests
docker compose --profile verification run --rm tests ruff check .
docker compose --profile verification run --rm tests ruff format --check .
docker compose --profile verification run --build --rm vision-worker
docker compose --profile verification run --build --rm temporal-worker
```

Tests create random schemas, apply migrations, exercise constraints and repositories,
then drop only those disposable schemas. Upgrade/downgrade tests never downgrade
the application schema. PostgreSQL integration tests require the local test role to
create schemas. Worker commands check infrastructure and exit; they do not consume
observations, acknowledge offsets or perform inference.

## Persistence, Backup and Shutdown

PostgreSQL and Kafka use named volumes. `docker compose down` removes containers
and the project network while retaining those volumes. `docker compose down -v`
deletes data and is not a routine restart command. Changing the environment password
does not rotate credentials already stored in an initialized PostgreSQL volume.

Create a private database backup before schema changes:

```bash
mkdir -p outputs/backups
umask 077
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > outputs/backups/shopper-analytics.dump
```

`outputs/` is ignored by Git. Check the backup's contents before relying on it:

```bash
docker compose exec -T postgres pg_restore --list < outputs/backups/shopper-analytics.dump
```

Restore rehearsal must target a separate disposable database, using the matching
PostgreSQL version. Application downgrade drops the domain tables; it is destructive
and belongs only to a reviewed rollback/recovery operation. The automated test suite
proves downgrade on isolated schemas, not restoration of production history.

## Review Checklist

- Compare the eleven ORM models with the complete migration chain and its constraints.
- Review required/null JSONB semantics, model dependencies and polygon bounds.
- Review snapshot history/cutover behavior and the concurrent editor test.
- Verify observation identity, natural uniqueness and persistence-before-offset ordering.
- Confirm retention manifests and raw-pruning eligibility before enabling deletion.
- Review deployment-specific access, transport, retention and inference policy before
  real-store ingestion; the current services expose only the local health API.

Implementation references: [Alembic async migrations](https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic),
[SQLAlchemy PostgreSQL types and indexes](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html),
and [SQLAlchemy async session lifecycle](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html).
