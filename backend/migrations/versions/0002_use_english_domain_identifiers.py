"""Use English domain identifiers and observation contract version 2.0.

Revision ID: 0002_english_names
Revises: 0001_data_layer

Legacy identifiers are retained only for reversible migration compatibility.
"""

import json
import re

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql.elements import conv

revision = "0002_english_names"
down_revision = "0001_data_layer"
branch_labels = None
depends_on = None

COLUMN_RENAMES = {
    "camera": ("camera", [("nome", "name"), ("ativa", "is_active")]),
    "versao_modelo": (
        "model_version",
        [
            ("nome", "name"),
            ("tipo", "kind"),
            ("versao", "version"),
            ("configuracao", "configuration"),
        ],
    ),
    "gondola": (
        "shelf",
        [
            ("nome", "name"),
            ("categoria", "category"),
            ("roi_produtos", "product_roi"),
            ("roi_presenca", "presence_roi"),
            ("ativa", "is_active"),
        ],
    ),
    "interacao_gondola": (
        "shelf_interaction",
        [
            ("gondola_id", "shelf_id"),
            ("inicio", "started_at"),
            ("fim", "ended_at"),
            ("ultima_presenca_em", "last_presence_at"),
            ("tempo_presenca_ms", "presence_duration_ms"),
            ("motivo_fim", "end_reason"),
        ],
    ),
    "metrica_gondola": (
        "shelf_metric",
        [
            ("janela_fim", "window_end"),
            ("qtd_interacoes_iniciadas", "interactions_started"),
            ("qtd_interacoes_com_atencao", "interactions_with_attention"),
            ("qtd_episodios_atencao", "attention_episodes"),
            ("tempo_presenca_ms", "presence_duration_ms"),
            ("tempo_atencao_ms", "attention_duration_ms"),
            ("qtd_alcances", "reach_count"),
            ("qtd_toques", "touch_count"),
            ("qtd_retiradas", "pickup_count"),
            ("qtd_devolucoes", "return_count"),
            ("gondola_id", "shelf_id"),
            ("granularidade", "granularity"),
            ("janela_inicio", "window_start"),
            ("atualizado_em", "updated_at"),
        ],
    ),
    "metrica_sinal_gondola": (
        "shelf_signal_metric",
        [
            ("tipo_sinal", "signal_type"),
            ("qtd_sinais", "signal_count"),
            ("gondola_id", "shelf_id"),
            ("granularidade", "granularity"),
            ("janela_inicio", "window_start"),
            ("atualizado_em", "updated_at"),
        ],
    ),
    "observacao_frame": (
        "frame_observation",
        [
            ("gondola_olhada_id", "attention_shelf_id"),
            ("gondola_alcancada_id", "reach_shelf_id"),
            ("track_id_local", "local_track_id"),
            ("observado_em", "observed_at"),
            ("atributos", "attributes"),
        ],
    ),
    "episodio_atencao": (
        "attention_episode",
        [
            ("interacao_id", "interaction_id"),
            ("modelo_id", "model_id"),
            ("inicio", "started_at"),
            ("fim", "ended_at"),
            ("metodo", "method"),
            ("confianca", "confidence"),
        ],
    ),
    "evento_acao": (
        "action_event",
        [
            ("interacao_id", "interaction_id"),
            ("modelo_id", "model_id"),
            ("tipo_acao", "action_type"),
            ("ocorrido_em", "occurred_at"),
            ("confianca", "confidence"),
            ("atributos", "attributes"),
        ],
    ),
    "observacao_interacao": (
        "observation_interaction",
        [
            ("observacao_id", "observation_id"),
            ("interacao_id", "interaction_id"),
            ("tipo_vinculo", "link_type"),
        ],
    ),
    "sinal_reacao": (
        "reaction_signal",
        [
            ("interacao_id", "interaction_id"),
            ("modelo_id", "model_id"),
            ("tipo_sinal", "signal_type"),
            ("origem", "source"),
            ("ocorrido_em", "occurred_at"),
            ("confianca", "confidence"),
            ("atributos", "attributes"),
        ],
    ),
}

DATABASE_NAMES = (
    "ix_versao_modelo_configuracao_gin",
    "ix_gondola_roi_presenca_gin",
    "ix_gondola_roi_produtos_gin",
    "ix_interacao_gondola_gondola_id_inicio",
    "ix_observacao_frame_atributos_gin",
    "ix_observacao_frame_camera_id_observado_em",
    "ix_observacao_frame_gondola_alcancada_id",
    "ix_observacao_frame_gondola_olhada_id",
    "ix_episodio_atencao_interacao_id_inicio",
    "ix_evento_acao_atributos_gin",
    "ix_evento_acao_interacao_id_tipo_acao_ocorrido_em",
    "ix_observacao_interacao_interacao_id",
    "ix_sinal_reacao_atributos_gin",
    "ix_sinal_reacao_interacao_id_tipo_sinal_ocorrido_em",
    "ix_gondola_camera_id",
    "ix_episodio_atencao_modelo_id",
    "ix_evento_acao_modelo_id",
    "ix_sinal_reacao_modelo_id",
    "ck_versao_modelo_configuration_object",
    "ck_versao_modelo_valid_kind",
    "pk_versao_modelo",
    "uq_versao_modelo_nome_tipo_versao",
    "ck_gondola_roi_objects",
    "ck_gondola_positive_roi_version",
    "fk_gondola_camera_id_camera",
    "pk_gondola",
    "ck_interacao_gondola_valid_lifecycle",
    "ck_interacao_gondola_valid_close_reason",
    "ck_interacao_gondola_ordered_interval",
    "ck_interacao_gondola_nonnegative_presence",
    "ck_interacao_gondola_valid_presence_endpoint",
    "fk_interacao_gondola_gondola_id_gondola",
    "pk_interacao_gondola",
    "ck_metrica_gondola_ordered_window",
    "ck_metrica_gondola_nonempty_granularity",
    "ck_metrica_gondola_nonnegative_metrics",
    "fk_metrica_gondola_gondola_id_gondola",
    "pk_metrica_gondola",
    "ck_metrica_sinal_gondola_nonempty_dimensions",
    "ck_metrica_sinal_gondola_nonnegative_count",
    "fk_metrica_sinal_gondola_gondola_id_gondola",
    "pk_metrica_sinal_gondola",
    "ck_observacao_frame_required_attribute_groups",
    "ck_observacao_frame_valid_frame_identity",
    "fk_observacao_frame_camera_id_camera",
    "fk_observacao_frame_gondola_alcancada_id_gondola",
    "fk_observacao_frame_gondola_olhada_id_gondola",
    "pk_observacao_frame",
    "uq_observacao_frame_camera_id_tracking_session_id_track_id_local_frame_seq",
    "ck_episodio_atencao_valid_confidence",
    "ck_episodio_atencao_ordered_interval",
    "fk_episodio_atencao_interacao_id_interacao_gondola",
    "fk_episodio_atencao_modelo_id_versao_modelo",
    "pk_episodio_atencao",
    "ck_evento_acao_required_attribute_groups",
    "ck_evento_acao_valid_action_type",
    "ck_evento_acao_valid_confidence",
    "fk_evento_acao_interacao_id_interacao_gondola",
    "fk_evento_acao_modelo_id_versao_modelo",
    "pk_evento_acao",
    "ck_observacao_interacao_valid_link_type",
    "fk_observacao_interacao_interacao_id_interacao_gondola",
    "fk_observacao_interacao_observacao_id_observacao_frame",
    "pk_observacao_interacao",
    "ck_sinal_reacao_required_attribute_groups",
    "ck_sinal_reacao_valid_reaction_source",
    "ck_sinal_reacao_valid_confidence",
    "fk_sinal_reacao_interacao_id_interacao_gondola",
    "fk_sinal_reacao_modelo_id_versao_modelo",
    "pk_sinal_reacao",
)

GUARDS = (
    ("model_version", "immutable_model_version"),
    ("shelf", "geometry_cutover"),
    ("frame_observation", "observation_camera"),
    ("observation_interaction", "observation_link"),
    ("attention_episode", "business_model"),
    ("action_event", "business_model"),
    ("reaction_signal", "business_model"),
)
FUNCTIONS = (
    "reject_model_mutation",
    "validate_geometry_cutover",
    "validate_observation_camera",
    "validate_observation_link",
    "validate_business_model",
)
JSON_KEYS = {
    "gondola_id": "shelf_id",
    "gondola_confidence": "shelf_confidence",
    "single_gondola_context": "single_shelf_context",
}


def _translate_identifiers(statement: str, *, english: bool, embedded: bool = False) -> str:
    names = {old: new for old, (new, _) in COLUMN_RENAMES.items()}
    names.update({old: new for _, columns in COLUMN_RENAMES.values() for old, new in columns})
    names.update(JSON_KEYS)
    if not english:
        names = {new: old for old, new in names.items()}
    boundary = "A-Za-z0-9" if embedded else "A-Za-z0-9_"
    pattern = (
        rf"(?<![{boundary}])("
        + "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        + rf")(?![{boundary}])"
    )
    return re.sub(pattern, lambda match: names[match[0]], statement)


def _drop_guards(*, english: bool) -> None:
    for table, trigger in GUARDS:
        if not english:
            table = _translate_identifiers(table, english=False)
        op.execute(f"DROP TRIGGER {trigger} ON {table}")
    for function in FUNCTIONS:
        op.execute(f"DROP FUNCTION {function}()")


def _rename_schema(*, english: bool) -> None:
    connection = op.get_bind()
    quote = connection.dialect.identifier_preparer.quote
    renames = COLUMN_RENAMES.items() if english else reversed(list(COLUMN_RENAMES.items()))
    for legacy_table, (new_table, columns) in renames:
        old, new = (legacy_table, new_table) if english else (new_table, legacy_table)
        if old != new:
            op.rename_table(old, new)
        for old_column, new_column in columns:
            if not english:
                old_column, new_column = new_column, old_column
            op.alter_column(new, old_column, new_column_name=new_column)
    preparer = connection.dialect.identifier_preparer
    renames = {}
    for old in DATABASE_NAMES:
        new = _translate_identifiers(old, english=True, embedded=True)
        old = preparer.truncate_and_render_constraint_name(conv(old), _alembic_quote=False)
        new = preparer.truncate_and_render_constraint_name(conv(new), _alembic_quote=False)
        renames[old if english else new] = new if english else old
    for legacy_table, (new_table, _) in COLUMN_RENAMES.items():
        table = new_table if english else legacy_table
        inspector = sa.inspect(connection)
        constraints = [
            inspector.get_pk_constraint(table),
            *inspector.get_unique_constraints(table),
            *inspector.get_foreign_keys(table),
            *inspector.get_check_constraints(table),
        ]
        for constraint in constraints:
            old = constraint["name"]
            if old in renames:
                op.execute(
                    f"ALTER TABLE {quote(table)} RENAME CONSTRAINT "
                    f"{quote(old)} TO {quote(renames[old])}"
                )
        for index in sa.inspect(connection).get_indexes(table):
            old = index["name"]
            if old in renames:
                op.execute(f"ALTER INDEX {quote(old)} RENAME TO {quote(renames[old])}")


def _migrate_contracts(*, english: bool) -> None:
    # Recurse only through approved structured JSONB; model parameters are opaque.
    op.execute("""
        CREATE FUNCTION migrate_contract_keys(document jsonb, names jsonb)
        RETURNS jsonb LANGUAGE plpgsql AS $$
        DECLARE result jsonb; item record; destination text;
        BEGIN
            IF jsonb_typeof(document) = 'object' THEN
                result := '{}'::jsonb;
                FOR item IN SELECT * FROM jsonb_each(document) LOOP
                    destination := COALESCE(names->>item.key, item.key);
                    IF result ? destination THEN
                        RAISE EXCEPTION 'Conflicting contract keys during naming migration'
                            USING ERRCODE = '23514';
                    END IF;
                    result := result || jsonb_build_object(destination,
                        migrate_contract_keys(item.value, names));
                END LOOP;
                RETURN result;
            ELSIF jsonb_typeof(document) = 'array' THEN
                RETURN COALESCE((SELECT jsonb_agg(migrate_contract_keys(value, names)
                    ORDER BY ordinality) FROM jsonb_array_elements(document)
                    WITH ORDINALITY), '[]'::jsonb);
            END IF;
            RETURN document;
        END $$
    """)
    names = JSON_KEYS if english else {new: old for old, new in JSON_KEYS.items()}
    mapping = json.dumps(names)
    for table, column in (
        ("shelf", "product_roi"),
        ("frame_observation", "attributes"),
        ("reaction_signal", "attributes"),
    ):
        op.get_bind().execute(
            sa.text(
                f"UPDATE {table} SET {column} = "
                f"migrate_contract_keys({column}, CAST(:names AS jsonb))"
            ),
            {"names": mapping},
        )
    old, new = ("1.0", "2.0") if english else ("2.0", "1.0")
    op.get_bind().execute(
        sa.text("""
        UPDATE frame_observation SET attributes = jsonb_set(attributes,
            '{provenance,schema_version}', to_jsonb(CAST(:new AS text)))
        WHERE attributes->'provenance'->>'schema_version' = :old
    """),
        {"new": new, "old": old},
    )
    op.execute("DROP FUNCTION migrate_contract_keys(jsonb, jsonb)")


def upgrade() -> None:
    _drop_guards(english=False)
    _rename_schema(english=True)
    _migrate_contracts(english=True)
    _install_guards(english=True)


def downgrade() -> None:
    _drop_guards(english=True)
    _migrate_contracts(english=False)
    _rename_schema(english=False)
    _install_guards(english=False)


def _install_guards(*, english: bool) -> None:
    def execute(statement: str) -> None:
        op.execute(statement if english else _translate_identifiers(statement, english=False))

    execute("""
        CREATE FUNCTION reject_model_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'Model versions are immutable; register a new version'
                USING ERRCODE = '23514';
        END $$
    """)
    execute("""
        CREATE TRIGGER immutable_model_version BEFORE UPDATE OR DELETE ON model_version
        FOR EACH ROW EXECUTE FUNCTION reject_model_mutation()
    """)
    execute("""
        CREATE FUNCTION validate_geometry_cutover() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE archived jsonb; old_history jsonb; new_history jsonb;
        BEGIN
            IF NEW.camera_id <> OLD.camera_id THEN
                RAISE EXCEPTION 'A shelf cannot change cameras' USING ERRCODE = '23514';
            END IF;
            IF NEW.product_roi = OLD.product_roi AND NEW.presence_roi = OLD.presence_roi
                AND NEW.roi_version = OLD.roi_version THEN
                RETURN NEW;
            END IF;
            IF EXISTS (SELECT 1 FROM shelf_interaction
                       WHERE shelf_id = OLD.id AND status = 'ACTIVE') THEN
                RAISE EXCEPTION 'Close active interactions before geometry cutover'
                    USING ERRCODE = '23514';
            END IF;
            old_history := COALESCE(OLD.product_roi->'history', '[]'::jsonb);
            new_history := NEW.product_roi->'history';
            IF NEW.roi_version <> OLD.roi_version + 1 OR new_history IS NULL
                OR jsonb_array_length(new_history) <> NEW.roi_version - 1
                OR new_history - (jsonb_array_length(new_history) - 1) <> old_history THEN
                RAISE EXCEPTION 'Geometry history must append exactly one previous version'
                    USING ERRCODE = '23514';
            END IF;
            archived := new_history->-1;
            IF (archived->>'roi_version')::integer IS DISTINCT FROM OLD.roi_version
                OR (archived->>'shelf_id')::uuid IS DISTINCT FROM OLD.id
                OR (archived->>'camera_id')::uuid IS DISTINCT FROM OLD.camera_id
                OR archived->'product_roi' IS DISTINCT FROM
                    OLD.product_roi - ARRAY['history','reference_revision']::text[]
                OR archived->'presence_roi' IS DISTINCT FROM OLD.presence_roi
                OR archived->>'reference_revision' IS DISTINCT FROM
                    OLD.product_roi->>'reference_revision'
                OR (archived->>'valid_from')::timestamptz IS DISTINCT FROM
                    COALESCE((old_history->-1->>'valid_until')::timestamptz, OLD.created_at)
                OR (archived->>'valid_until')::timestamptz IS NULL
                OR (archived->>'valid_until')::timestamptz <=
                    (archived->>'valid_from')::timestamptz THEN
                RAISE EXCEPTION 'Archived geometry must preserve the previous accepted snapshot'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$
    """)
    execute("""
        CREATE TRIGGER geometry_cutover BEFORE UPDATE ON shelf
        FOR EACH ROW EXECUTE FUNCTION validate_geometry_cutover()
    """)
    execute("""
        CREATE FUNCTION validate_observation_camera() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE target uuid;
        BEGIN
            FOREACH target IN ARRAY ARRAY[NEW.attention_shelf_id, NEW.reach_shelf_id] LOOP
                IF target IS NOT NULL AND NOT EXISTS
                    (SELECT 1 FROM shelf WHERE id = target AND camera_id = NEW.camera_id) THEN
                    RAISE EXCEPTION 'Observation target belongs to a different or absent camera'
                        USING ERRCODE = '23514';
                END IF;
            END LOOP;
            RETURN NEW;
        END $$
    """)
    execute("""
        CREATE TRIGGER observation_camera BEFORE INSERT OR UPDATE ON frame_observation
        FOR EACH ROW EXECUTE FUNCTION validate_observation_camera()
    """)
    execute("""
        CREATE FUNCTION validate_observation_link() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE observation_camera uuid; interaction_camera uuid; target uuid;
        BEGIN
            SELECT camera_id INTO observation_camera FROM frame_observation
                WHERE id = NEW.observation_id;
            SELECT g.camera_id, g.id INTO interaction_camera, target
                FROM shelf_interaction i JOIN shelf g ON g.id = i.shelf_id
                WHERE i.id = NEW.interaction_id;
            IF observation_camera IS NOT NULL AND interaction_camera IS NOT NULL
                AND observation_camera <> interaction_camera THEN
                RAISE EXCEPTION 'Observation links cannot cross camera scope'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$
    """)
    execute("""
        CREATE TRIGGER observation_link BEFORE INSERT OR UPDATE ON observation_interaction
        FOR EACH ROW EXECUTE FUNCTION validate_observation_link()
    """)
    execute("""
        CREATE FUNCTION validate_business_model() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE expected text; actual text;
        BEGIN
            IF TG_TABLE_NAME = 'attention_episode' THEN
                expected := 'ATTENTION';
            ELSIF TG_TABLE_NAME = 'action_event' THEN
                expected := 'ACTION';
            ELSE
                expected := CASE NEW.source WHEN 'FACE_AND_BODY' THEN 'FACE_BODY_REACTION'
                    ELSE 'BODY_REACTION' END;
            END IF;
            SELECT kind INTO actual FROM model_version WHERE id = NEW.model_id;
            IF actual IS NOT NULL AND actual <> expected THEN
                RAISE EXCEPTION 'Model kind is incompatible with business evidence source'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$
    """)
    for table in ("attention_episode", "action_event", "reaction_signal"):
        execute(
            f"CREATE TRIGGER business_model BEFORE INSERT OR UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION validate_business_model()"
        )
