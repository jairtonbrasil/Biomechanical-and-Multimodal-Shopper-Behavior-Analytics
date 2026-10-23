# Persistence and Data Contracts

The version 2.0 contracts preserve the eleven-table domain schema. Physical table
and column identifiers, Python contracts, JSON keys and technical documentation
use English. `alembic_version` is migration metadata, not
an additional domain entity.

Revision `0002_english_names` standardizes the physical schema and nested JSONB
keys in English while preserving rows, UUIDs, relationships and ROI history.
Observation contract **2.0** is a breaking naming change: producers must use
`attributes`, `local_track_id`, `observed_at`, `attention_shelf_id`, `reach_shelf_id`
and nested shelf references. Version 1.0 messages and legacy field aliases are
rejected by the current schemas. Stored observations are converted by the migration.
Model configuration retains its independent version 1.0; UUIDv5 identity encoding
also retains version 1.0 so existing identities do not change.

## Observation Envelope

`app/schemas/observation.py` defines `ObservationEnvelope` with exactly two top-level
keys: `schema_version: "2.0"` and `observation`. Provenance is persisted once under
`observation.attributes.provenance`; there is no duplicate top-level provenance
object and no unpersisted metadata needed to interpret a stored observation.

The observation includes its UUID, camera UUID, optional attention/reach shelves,
tracking session UUID, local track identifier, frame sequence, event timestamp,
and `attributes`. Timestamps must include a timezone. Unknown optional targets are
explicitly null. Unknown contract versions and unrecognized fields are rejected.

| JSONB group | Contract |
|---|---|
| `bbox` | Normalized bounds and confidence, or null when unavailable; positive dimensions when present |
| `pose` | All 17 COCO keypoint names; each is a normalized x/y/confidence measurement or null |
| `head_pose` | Nullable yaw/pitch/roll and quality; method `FACE`, `BODY_FALLBACK`, or `UNAVAILABLE` |
| `attention` | Nullable direction detection, diagnostic focus duration and shelf confidence |
| `hands` | Left/right normalized coordinates and nullable product ROI membership |
| `movement` | `STANDING`, `WALKING`, or null; nonnegative measured velocity or null |
| `face` | Nullable visibility, expression and confidence; no expression for an unobserved face |
| `provenance` | Contract version, pose/optional-face/rule UUIDs, source epoch, timestamp source, source/sample FPS and image dimensions |
| `geometry` | Full immutable snapshots of every evaluated shelf/ROI version |
| `presence` | Evaluated shelf memberships, anchor measurements and quality |
| `kinematics` | Usable shoulder width in pixels, per-arm angles/reach ratios, normalized motion, bimanual/self-adaptor/yaw indicators, window size and nullable candidates |
| `spatial` | Nullable foot, hand, head-origin and gaze-target points with confidence |
| `evidence` | Product evidence provider, capability availability and nullable observable object evidence |

Required groups and their required nested keys remain present even when their
measurements are null. `null` means unavailable/unknown; `false` means a supported
negative observation. An invisible face is not neutral. Invalid normalization
scale cannot produce kinematic candidates. Numeric values must be finite; no
image, video, face crop, camera credential or personal identity is part of this
contract. Actual calibration and inference acceptance belong to the vision stage.

The envelope budget is 256 KiB minus a 512-byte transport allowance. Geometry is
bounded to 32 evaluated shelves per observation. Polygon validation accepts 3–64
distinct normalized vertices, rejects intersections and numerically degenerate
area, and requires an open ring without a repeated final vertex. Clockwise and
counterclockwise polygons are accepted. Coordinates are normalized for storage;
biomechanical distances must use the aspect-corrected image plane.

## Geometry History and Cutover

The chosen provenance design uses **full snapshots**, not references to mutable
current geometry. `shelf.product_roi` contains the current polygon,
`reference_revision`, and an append-only `history` array. `presence_roi` contains
the current presence polygon. Each archived snapshot includes camera/shelf
UUIDs, ROI version, reference revision, both polygons and a half-open validity
interval. No history table or extra SQL column is introduced.

`ShelfRepository.update_geometry` locks the row, checks `expected_version`,
preserves the old snapshot, advances the version, and flushes within the caller's
transaction. A second editor using the stale version fails. Database triggers
also reject camera reassignment, history rewrites, skipped versions and cutovers
while active interactions exist. Later operational services must close old-version
interactions in the same transaction before performing a cutover.

An observation carries only the snapshots evaluated in its frame, not the whole
shelf history. A delayed observation can use an earlier version only within its
registered event-time interval. An originally open-ended snapshot remains
interpretable after the registered interval is closed. Historical snapshots are
bounded to 256 previous versions; exceeding the bound fails instead of silently
discarding reconstruction evidence. Reference-image capture/storage is a separate
operational feature; the reference UUID here establishes identity, not image storage.

The synthetic sizing tests compare a minimum-group envelope with its full
provenance extensions, using compact UTF-8 JSON and PostgreSQL `pg_column_size` of
the corresponding JSONB datum. Run them with:

```bash
docker compose --profile verification run --build --rm tests \
  pytest tests/test_contract_sizes.py -s -q -p no:cacheprovider
```

These measurements describe one synthetic observation with one four-vertex
shelf. They do not estimate daily storage: actual camera count, frame rate,
visibility, evaluated ROIs, indexes, tuple overhead, TOAST compression and retention
must be benchmarked. Full snapshots deliberately trade payload size for independent
historical interpretation.

The sizing test prints baseline, full-envelope and extension sizes for the current
fixture and contract version. JSONB datum measurements exclude table/index/TOAST
layout and are not a disk-capacity forecast.

## Immutable Model Versions

`model_version` has a unique `(name, kind, version)` and a validated `configuration`:
schema version, optional artifact name/SHA-256, parameters and versioned dependencies.
Neural models require artifact checksums. URLs and credential-bearing parameter
keys are rejected. Registering identical content is idempotent; changed content
requires a new version. PostgreSQL rejects updates and deletes of registered versions.

| Model kind | Use / required dependencies |
|---|---|
| `POSE` | Immutable pose artifact |
| `FACE` | Immutable facial artifact |
| `TEMPORAL_RULES` | Versioned temporal parameters |
| `ATTENTION` | Pose + temporal rules |
| `ACTION` | Pose + temporal rules |
| `BODY_REACTION` | Pose + temporal rules |
| `FACE_BODY_REACTION` | Pose + face + temporal rules |

Repositories verify that referenced dependency UUIDs exist with their declared
kind. Attention episodes use `ATTENTION`, actions use `ACTION`, and reaction model
kinds match `FACE_AND_BODY` or `BODY_BIOMECHANICS`. Database triggers protect these
business model/source associations as well. Model confidence is not measured accuracy.

## Business Records and Identity

`app/domain/identity.py` uses versioned UUIDv5 identities with canonical JSON tuple
encoding. Observation identity includes camera/session/local-track/frame. Interaction
identity adds shelf, geometry version, first qualifying frame and rule version.
Event identity includes its parent interaction, event type, first qualifying frame
and model version. Type and frame boundaries distinguish bouts; changed rules produce
different identities. A natural database uniqueness constraint additionally protects
the scoped observation frame. Reusing an observation ID with changed content fails.
These primitives support replay; they do not implement the temporal recovery engine.

Active interactions have null `ended_at` and `end_reason`; closed interactions require
both. Intervals and presence endpoints are ordered, durations are nonnegative, and
attention episodes may remain open. Observation links have the triple primary key
`(observation_id, interaction_id, link_type)` with `PRESENCE`, `ATTENTION`, or `REACH`.
Repositories validate link evidence and geometry; database triggers reject
cross-camera targets and links.

Action types remain exactly `REACH`, `INSPECT`, `PICKUP`, and `RETURN`. `hand` remains
`left` or `right`; for bimanual evidence it identifies the initiating/representative
hand and `bimanual` is true. Product class, identifier and SKU may be null. A pickup
or return requires provider-backed object displacement evidence; pose/ROI contact
alone is insufficient. The JSON key `class` is preserved through an English Python
field alias. No `TOUCH` enum is added or aliased to pickup.

Reactions preserve facial analysis, behavioral context, temporal evidence,
association and optional inferred emotion. Body-only signals cannot manufacture
facial labels or emotions. Facial/body fusion requires usable facial evidence.

Metric primary keys follow the supplied schema. Counts/durations are nonnegative;
shelf metric windows have positive duration. Supported granularities and evidence
coverage remain analytical-service policy; a stored zero alone does not establish
that a detector supported a capability.

## Retention and Reconstruction Eligibility

All foreign keys use `ON DELETE RESTRICT`. Deleting cameras, shelves, models or
raw observations cannot cascade away business records. The tested raw-pruning order
is to explicitly remove eligible observation links, then their raw observations;
interactions, events, model versions and aggregate rows remain intact. That operation
reduces track ownership and replay evidence even when business rows survive.

No automatic deletion policy is enabled. A deployment must agree retention and
invalidate/update coverage before pruning. `HistoryRepository.assess` fails closed
without a trusted `VerifiedCoverage` manifest for the requested camera/window.
Raw coverage is required for replay and unique tracks; unique tracks additionally
require complete observation-link coverage. Business coverage is distinct and used
for reaggregation. Replay extends its required start to include active interactions
and validates retained model/geometry provenance in bounded batches.

The eligibility layer consumes a verified operational manifest; it does not create
or persist a coverage certificate, infer completeness from the oldest surviving
row, detect all capture gaps, or execute replay. The future ingest/retention workflow
must issue and maintain those manifests. A verified empty interval is different
from unverified missing data. Deployment retention lifetimes, processing policy and
recovery capacity remain operational decisions.
