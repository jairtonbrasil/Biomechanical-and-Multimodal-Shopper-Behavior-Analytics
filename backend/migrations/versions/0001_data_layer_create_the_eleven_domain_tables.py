"""Create the eleven domain tables

Revision ID: 0001_data_layer
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_data_layer"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "camera",
        sa.Column("nome", sa.String(length=128), nullable=False),
        sa.Column("stream_url", sa.Text(), nullable=False),
        sa.Column("ativa", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_camera")),
    )
    op.create_table(
        "versao_modelo",
        sa.Column("nome", sa.String(length=128), nullable=False),
        sa.Column("tipo", sa.String(length=32), nullable=False),
        sa.Column("versao", sa.String(length=128), nullable=False),
        sa.Column("configuracao", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(configuracao) = 'object'",
            name=op.f("ck_versao_modelo_configuration_object"),
        ),
        sa.CheckConstraint(
            (
                "tipo IN "
                "('POSE','FACE','TEMPORAL_RULES','ATTENTION','ACTION','BODY_REACTION','FACE_BODY_REACTION')"
            ),
            name=op.f("ck_versao_modelo_valid_kind"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_versao_modelo")),
        sa.UniqueConstraint(
            "nome", "tipo", "versao", name=op.f("uq_versao_modelo_nome_tipo_versao")
        ),
    )
    op.create_index(
        "ix_versao_modelo_configuracao_gin",
        "versao_modelo",
        ["configuracao"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "gondola",
        sa.Column("camera_id", sa.Uuid(), nullable=False),
        sa.Column("nome", sa.String(length=128), nullable=False),
        sa.Column("categoria", sa.String(length=128), nullable=False),
        sa.Column("roi_produtos", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("roi_presenca", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("roi_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("ativa", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "jsonb_typeof(roi_produtos) = 'object' AND jsonb_typeof(roi_presenca) = 'object'",
            name=op.f("ck_gondola_roi_objects"),
        ),
        sa.CheckConstraint("roi_version >= 1", name=op.f("ck_gondola_positive_roi_version")),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["camera.id"],
            name=op.f("fk_gondola_camera_id_camera"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_gondola")),
    )
    op.create_index(op.f("ix_gondola_camera_id"), "gondola", ["camera_id"], unique=False)
    op.create_index(
        "ix_gondola_roi_presenca_gin",
        "gondola",
        ["roi_presenca"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_gondola_roi_produtos_gin",
        "gondola",
        ["roi_produtos"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_table(
        "interacao_gondola",
        sa.Column("gondola_id", sa.Uuid(), nullable=False),
        sa.Column("inicio", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fim", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ultima_presenca_em", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tempo_presenca_ms", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("motivo_fim", sa.String(length=32), nullable=True),
        sa.Column("roi_version", sa.Integer(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "(status = 'ACTIVE' AND fim IS NULL AND motivo_fim IS NULL) OR "
                "(status = 'CLOSED' AND fim IS NOT NULL AND motivo_fim IS NOT NULL)"
            ),
            name=op.f("ck_interacao_gondola_valid_lifecycle"),
        ),
        sa.CheckConstraint(
            (
                "motivo_fim IS NULL OR motivo_fim IN "
                "('LEFT_ROI','TRACK_LOST','CAMERA_OFFLINE','ROI_CHANGED','REPROCESSING_GAP')"
            ),
            name=op.f("ck_interacao_gondola_valid_close_reason"),
        ),
        sa.CheckConstraint(
            "fim IS NULL OR fim >= inicio", name=op.f("ck_interacao_gondola_ordered_interval")
        ),
        sa.CheckConstraint(
            "roi_version >= 1 AND tempo_presenca_ms >= 0",
            name=op.f("ck_interacao_gondola_nonnegative_presence"),
        ),
        sa.CheckConstraint(
            "ultima_presenca_em >= inicio AND (fim IS NULL OR ultima_presenca_em <= fim)",
            name=op.f("ck_interacao_gondola_valid_presence_endpoint"),
        ),
        sa.ForeignKeyConstraint(
            ["gondola_id"],
            ["gondola.id"],
            name=op.f("fk_interacao_gondola_gondola_id_gondola"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_interacao_gondola")),
    )
    op.create_index(
        "ix_interacao_gondola_gondola_id_inicio",
        "interacao_gondola",
        ["gondola_id", "inicio"],
        unique=False,
    )
    op.create_table(
        "metrica_gondola",
        sa.Column("janela_fim", sa.DateTime(timezone=True), nullable=False),
        sa.Column("qtd_interacoes_iniciadas", sa.Integer(), server_default="0", nullable=False),
        sa.Column("qtd_interacoes_com_atencao", sa.Integer(), server_default="0", nullable=False),
        sa.Column("qtd_episodios_atencao", sa.Integer(), server_default="0", nullable=False),
        sa.Column("tempo_presenca_ms", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("tempo_atencao_ms", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("qtd_alcances", sa.Integer(), server_default="0", nullable=False),
        sa.Column("qtd_toques", sa.Integer(), server_default="0", nullable=False),
        sa.Column("qtd_retiradas", sa.Integer(), server_default="0", nullable=False),
        sa.Column("qtd_devolucoes", sa.Integer(), server_default="0", nullable=False),
        sa.Column("gondola_id", sa.Uuid(), nullable=False),
        sa.Column("granularidade", sa.String(length=32), nullable=False),
        sa.Column("janela_inicio", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "atualizado_em",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "janela_fim > janela_inicio", name=op.f("ck_metrica_gondola_ordered_window")
        ),
        sa.CheckConstraint(
            "length(granularidade) > 0", name=op.f("ck_metrica_gondola_nonempty_granularity")
        ),
        sa.CheckConstraint(
            (
                "qtd_interacoes_iniciadas >= 0 AND qtd_interacoes_com_atencao >= 0 "
                "AND qtd_episodios_atencao >= 0 AND tempo_presenca_ms >= 0 AND "
                "tempo_atencao_ms >= 0 AND qtd_alcances >= 0 AND qtd_toques >= 0 AND "
                "qtd_retiradas >= 0 AND qtd_devolucoes >= 0"
            ),
            name=op.f("ck_metrica_gondola_nonnegative_metrics"),
        ),
        sa.ForeignKeyConstraint(
            ["gondola_id"],
            ["gondola.id"],
            name=op.f("fk_metrica_gondola_gondola_id_gondola"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "gondola_id", "granularidade", "janela_inicio", name=op.f("pk_metrica_gondola")
        ),
    )
    op.create_table(
        "metrica_sinal_gondola",
        sa.Column("tipo_sinal", sa.String(length=64), nullable=False),
        sa.Column("qtd_sinais", sa.Integer(), server_default="0", nullable=False),
        sa.Column("gondola_id", sa.Uuid(), nullable=False),
        sa.Column("granularidade", sa.String(length=32), nullable=False),
        sa.Column("janela_inicio", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "atualizado_em",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "length(granularidade) > 0 AND length(tipo_sinal) > 0",
            name=op.f("ck_metrica_sinal_gondola_nonempty_dimensions"),
        ),
        sa.CheckConstraint(
            "qtd_sinais >= 0", name=op.f("ck_metrica_sinal_gondola_nonnegative_count")
        ),
        sa.ForeignKeyConstraint(
            ["gondola_id"],
            ["gondola.id"],
            name=op.f("fk_metrica_sinal_gondola_gondola_id_gondola"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "tipo_sinal",
            "gondola_id",
            "granularidade",
            "janela_inicio",
            name=op.f("pk_metrica_sinal_gondola"),
        ),
    )
    op.create_table(
        "observacao_frame",
        sa.Column("camera_id", sa.Uuid(), nullable=False),
        sa.Column("gondola_olhada_id", sa.Uuid(), nullable=True),
        sa.Column("gondola_alcancada_id", sa.Uuid(), nullable=True),
        sa.Column("tracking_session_id", sa.Uuid(), nullable=False),
        sa.Column("track_id_local", sa.String(length=128), nullable=False),
        sa.Column("frame_seq", sa.BigInteger(), nullable=False),
        sa.Column("observado_em", sa.DateTime(timezone=True), nullable=False),
        sa.Column("atributos", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "jsonb_typeof(atributos) = 'object' AND atributos ?& "
                "ARRAY['bbox','pose','head_pose','attention','hands','movement','face','provenance','geometry','presence','kinematics','spatial','evidence']"
            ),
            name=op.f("ck_observacao_frame_required_attribute_groups"),
        ),
        sa.CheckConstraint(
            "frame_seq >= 0 AND length(track_id_local) > 0",
            name=op.f("ck_observacao_frame_valid_frame_identity"),
        ),
        sa.ForeignKeyConstraint(
            ["camera_id"],
            ["camera.id"],
            name=op.f("fk_observacao_frame_camera_id_camera"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["gondola_alcancada_id"],
            ["gondola.id"],
            name=op.f("fk_observacao_frame_gondola_alcancada_id_gondola"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["gondola_olhada_id"],
            ["gondola.id"],
            name=op.f("fk_observacao_frame_gondola_olhada_id_gondola"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_observacao_frame")),
        sa.UniqueConstraint(
            "camera_id",
            "tracking_session_id",
            "track_id_local",
            "frame_seq",
            name=op.f("uq_observacao_frame_camera_id_tracking_session_id_track_id_local_frame_seq"),
        ),
    )
    op.create_index(
        "ix_observacao_frame_atributos_gin",
        "observacao_frame",
        ["atributos"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_observacao_frame_camera_id_observado_em",
        "observacao_frame",
        ["camera_id", "observado_em"],
        unique=False,
    )
    op.create_index(
        "ix_observacao_frame_gondola_alcancada_id",
        "observacao_frame",
        ["gondola_alcancada_id"],
        unique=False,
    )
    op.create_index(
        "ix_observacao_frame_gondola_olhada_id",
        "observacao_frame",
        ["gondola_olhada_id"],
        unique=False,
    )
    op.create_table(
        "episodio_atencao",
        sa.Column("interacao_id", sa.Uuid(), nullable=False),
        sa.Column("modelo_id", sa.Uuid(), nullable=False),
        sa.Column("inicio", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fim", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metodo", sa.String(length=64), nullable=False),
        sa.Column("confianca", sa.REAL(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confianca >= 0 AND confianca <= 1", name=op.f("ck_episodio_atencao_valid_confidence")
        ),
        sa.CheckConstraint(
            "fim IS NULL OR fim >= inicio", name=op.f("ck_episodio_atencao_ordered_interval")
        ),
        sa.ForeignKeyConstraint(
            ["interacao_id"],
            ["interacao_gondola.id"],
            name=op.f("fk_episodio_atencao_interacao_id_interacao_gondola"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["modelo_id"],
            ["versao_modelo.id"],
            name=op.f("fk_episodio_atencao_modelo_id_versao_modelo"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_episodio_atencao")),
    )
    op.create_index(
        "ix_episodio_atencao_interacao_id_inicio",
        "episodio_atencao",
        ["interacao_id", "inicio"],
        unique=False,
    )
    op.create_index(
        op.f("ix_episodio_atencao_modelo_id"), "episodio_atencao", ["modelo_id"], unique=False
    )
    op.create_table(
        "evento_acao",
        sa.Column("interacao_id", sa.Uuid(), nullable=False),
        sa.Column("modelo_id", sa.Uuid(), nullable=False),
        sa.Column("tipo_acao", sa.String(length=16), nullable=False),
        sa.Column("ocorrido_em", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confianca", sa.REAL(), nullable=False),
        sa.Column("atributos", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "jsonb_typeof(atributos) = 'object' AND atributos ?& "
                "ARRAY['hand','detection_method','object','interaction','temporal_evidence']"
            ),
            name=op.f("ck_evento_acao_required_attribute_groups"),
        ),
        sa.CheckConstraint(
            "tipo_acao IN ('REACH','INSPECT','PICKUP','RETURN')",
            name=op.f("ck_evento_acao_valid_action_type"),
        ),
        sa.CheckConstraint(
            "confianca >= 0 AND confianca <= 1", name=op.f("ck_evento_acao_valid_confidence")
        ),
        sa.ForeignKeyConstraint(
            ["interacao_id"],
            ["interacao_gondola.id"],
            name=op.f("fk_evento_acao_interacao_id_interacao_gondola"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["modelo_id"],
            ["versao_modelo.id"],
            name=op.f("fk_evento_acao_modelo_id_versao_modelo"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evento_acao")),
    )
    op.create_index(
        "ix_evento_acao_atributos_gin",
        "evento_acao",
        ["atributos"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_evento_acao_interacao_id_tipo_acao_ocorrido_em",
        "evento_acao",
        ["interacao_id", "tipo_acao", "ocorrido_em"],
        unique=False,
    )
    op.create_index(op.f("ix_evento_acao_modelo_id"), "evento_acao", ["modelo_id"], unique=False)
    op.create_table(
        "observacao_interacao",
        sa.Column("observacao_id", sa.Uuid(), nullable=False),
        sa.Column("interacao_id", sa.Uuid(), nullable=False),
        sa.Column("tipo_vinculo", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "tipo_vinculo IN ('PRESENCE','ATTENTION','REACH')",
            name=op.f("ck_observacao_interacao_valid_link_type"),
        ),
        sa.ForeignKeyConstraint(
            ["interacao_id"],
            ["interacao_gondola.id"],
            name=op.f("fk_observacao_interacao_interacao_id_interacao_gondola"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["observacao_id"],
            ["observacao_frame.id"],
            name=op.f("fk_observacao_interacao_observacao_id_observacao_frame"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "observacao_id", "interacao_id", "tipo_vinculo", name=op.f("pk_observacao_interacao")
        ),
    )
    op.create_index(
        "ix_observacao_interacao_interacao_id",
        "observacao_interacao",
        ["interacao_id"],
        unique=False,
    )
    op.create_table(
        "sinal_reacao",
        sa.Column("interacao_id", sa.Uuid(), nullable=False),
        sa.Column("modelo_id", sa.Uuid(), nullable=False),
        sa.Column("tipo_sinal", sa.String(length=64), nullable=False),
        sa.Column("origem", sa.String(length=32), nullable=False),
        sa.Column("ocorrido_em", sa.DateTime(timezone=True), nullable=False),
        sa.Column("confianca", sa.REAL(), nullable=False),
        sa.Column("atributos", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "jsonb_typeof(atributos) = 'object' AND atributos ?& "
                "ARRAY['facial_analysis','behavioral_context','temporal_evidence','association','inferred_emotion']"
            ),
            name=op.f("ck_sinal_reacao_required_attribute_groups"),
        ),
        sa.CheckConstraint(
            "origem IN ('FACE_AND_BODY','BODY_BIOMECHANICS')",
            name=op.f("ck_sinal_reacao_valid_reaction_source"),
        ),
        sa.CheckConstraint(
            "confianca >= 0 AND confianca <= 1", name=op.f("ck_sinal_reacao_valid_confidence")
        ),
        sa.ForeignKeyConstraint(
            ["interacao_id"],
            ["interacao_gondola.id"],
            name=op.f("fk_sinal_reacao_interacao_id_interacao_gondola"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["modelo_id"],
            ["versao_modelo.id"],
            name=op.f("fk_sinal_reacao_modelo_id_versao_modelo"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_sinal_reacao")),
    )
    op.create_index(
        "ix_sinal_reacao_atributos_gin",
        "sinal_reacao",
        ["atributos"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "ix_sinal_reacao_interacao_id_tipo_sinal_ocorrido_em",
        "sinal_reacao",
        ["interacao_id", "tipo_sinal", "ocorrido_em"],
        unique=False,
    )
    op.create_index(op.f("ix_sinal_reacao_modelo_id"), "sinal_reacao", ["modelo_id"], unique=False)
    _install_guards()


def downgrade() -> None:
    _drop_guards()
    op.drop_index(op.f("ix_sinal_reacao_modelo_id"), table_name="sinal_reacao")
    op.drop_index("ix_sinal_reacao_interacao_id_tipo_sinal_ocorrido_em", table_name="sinal_reacao")
    op.drop_index(
        "ix_sinal_reacao_atributos_gin", table_name="sinal_reacao", postgresql_using="gin"
    )
    op.drop_table("sinal_reacao")
    op.drop_index("ix_observacao_interacao_interacao_id", table_name="observacao_interacao")
    op.drop_table("observacao_interacao")
    op.drop_index(op.f("ix_evento_acao_modelo_id"), table_name="evento_acao")
    op.drop_index("ix_evento_acao_interacao_id_tipo_acao_ocorrido_em", table_name="evento_acao")
    op.drop_index("ix_evento_acao_atributos_gin", table_name="evento_acao", postgresql_using="gin")
    op.drop_table("evento_acao")
    op.drop_index(op.f("ix_episodio_atencao_modelo_id"), table_name="episodio_atencao")
    op.drop_index("ix_episodio_atencao_interacao_id_inicio", table_name="episodio_atencao")
    op.drop_table("episodio_atencao")
    op.drop_index("ix_observacao_frame_gondola_olhada_id", table_name="observacao_frame")
    op.drop_index("ix_observacao_frame_gondola_alcancada_id", table_name="observacao_frame")
    op.drop_index("ix_observacao_frame_camera_id_observado_em", table_name="observacao_frame")
    op.drop_index(
        "ix_observacao_frame_atributos_gin", table_name="observacao_frame", postgresql_using="gin"
    )
    op.drop_table("observacao_frame")
    op.drop_table("metrica_sinal_gondola")
    op.drop_table("metrica_gondola")
    op.drop_index("ix_interacao_gondola_gondola_id_inicio", table_name="interacao_gondola")
    op.drop_table("interacao_gondola")
    op.drop_index("ix_gondola_roi_produtos_gin", table_name="gondola", postgresql_using="gin")
    op.drop_index("ix_gondola_roi_presenca_gin", table_name="gondola", postgresql_using="gin")
    op.drop_index(op.f("ix_gondola_camera_id"), table_name="gondola")
    op.drop_table("gondola")
    op.drop_index(
        "ix_versao_modelo_configuracao_gin", table_name="versao_modelo", postgresql_using="gin"
    )
    op.drop_table("versao_modelo")
    op.drop_table("camera")


def _install_guards() -> None:
    op.execute("""
        CREATE FUNCTION reject_model_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'Model versions are immutable; register a new version'
                USING ERRCODE = '23514';
        END $$
    """)
    op.execute("""
        CREATE TRIGGER immutable_model_version BEFORE UPDATE OR DELETE ON versao_modelo
        FOR EACH ROW EXECUTE FUNCTION reject_model_mutation()
    """)
    op.execute("""
        CREATE FUNCTION validate_geometry_cutover() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE archived jsonb; old_history jsonb; new_history jsonb;
        BEGIN
            IF NEW.camera_id <> OLD.camera_id THEN
                RAISE EXCEPTION 'A gondola cannot change cameras' USING ERRCODE = '23514';
            END IF;
            IF NEW.roi_produtos = OLD.roi_produtos AND NEW.roi_presenca = OLD.roi_presenca
                AND NEW.roi_version = OLD.roi_version THEN
                RETURN NEW;
            END IF;
            IF EXISTS (SELECT 1 FROM interacao_gondola
                       WHERE gondola_id = OLD.id AND status = 'ACTIVE') THEN
                RAISE EXCEPTION 'Close active interactions before geometry cutover'
                    USING ERRCODE = '23514';
            END IF;
            old_history := COALESCE(OLD.roi_produtos->'history', '[]'::jsonb);
            new_history := NEW.roi_produtos->'history';
            IF NEW.roi_version <> OLD.roi_version + 1 OR new_history IS NULL
                OR jsonb_array_length(new_history) <> NEW.roi_version - 1
                OR new_history - (jsonb_array_length(new_history) - 1) <> old_history THEN
                RAISE EXCEPTION 'Geometry history must append exactly one previous version'
                    USING ERRCODE = '23514';
            END IF;
            archived := new_history->-1;
            IF (archived->>'roi_version')::integer IS DISTINCT FROM OLD.roi_version
                OR (archived->>'gondola_id')::uuid IS DISTINCT FROM OLD.id
                OR (archived->>'camera_id')::uuid IS DISTINCT FROM OLD.camera_id
                OR archived->'product_roi' IS DISTINCT FROM
                    OLD.roi_produtos - ARRAY['history','reference_revision']::text[]
                OR archived->'presence_roi' IS DISTINCT FROM OLD.roi_presenca
                OR archived->>'reference_revision' IS DISTINCT FROM
                    OLD.roi_produtos->>'reference_revision'
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
    op.execute("""
        CREATE TRIGGER geometry_cutover BEFORE UPDATE ON gondola
        FOR EACH ROW EXECUTE FUNCTION validate_geometry_cutover()
    """)
    op.execute("""
        CREATE FUNCTION validate_observation_camera() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE target uuid;
        BEGIN
            FOREACH target IN ARRAY ARRAY[NEW.gondola_olhada_id, NEW.gondola_alcancada_id] LOOP
                IF target IS NOT NULL AND NOT EXISTS
                    (SELECT 1 FROM gondola WHERE id = target AND camera_id = NEW.camera_id) THEN
                    RAISE EXCEPTION 'Observation target belongs to a different or absent camera'
                        USING ERRCODE = '23514';
                END IF;
            END LOOP;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER observation_camera BEFORE INSERT OR UPDATE ON observacao_frame
        FOR EACH ROW EXECUTE FUNCTION validate_observation_camera()
    """)
    op.execute("""
        CREATE FUNCTION validate_observation_link() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE observation_camera uuid; interaction_camera uuid; target uuid;
        BEGIN
            SELECT camera_id INTO observation_camera FROM observacao_frame
                WHERE id = NEW.observacao_id;
            SELECT g.camera_id, g.id INTO interaction_camera, target
                FROM interacao_gondola i JOIN gondola g ON g.id = i.gondola_id
                WHERE i.id = NEW.interacao_id;
            IF observation_camera IS NOT NULL AND interaction_camera IS NOT NULL
                AND observation_camera <> interaction_camera THEN
                RAISE EXCEPTION 'Observation links cannot cross camera scope'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER observation_link BEFORE INSERT OR UPDATE ON observacao_interacao
        FOR EACH ROW EXECUTE FUNCTION validate_observation_link()
    """)
    op.execute("""
        CREATE FUNCTION validate_business_model() RETURNS trigger LANGUAGE plpgsql AS $$
        DECLARE expected text; actual text;
        BEGIN
            IF TG_TABLE_NAME = 'episodio_atencao' THEN
                expected := 'ATTENTION';
            ELSIF TG_TABLE_NAME = 'evento_acao' THEN
                expected := 'ACTION';
            ELSE
                expected := CASE NEW.origem WHEN 'FACE_AND_BODY' THEN 'FACE_BODY_REACTION'
                    ELSE 'BODY_REACTION' END;
            END IF;
            SELECT tipo INTO actual FROM versao_modelo WHERE id = NEW.modelo_id;
            IF actual IS NOT NULL AND actual <> expected THEN
                RAISE EXCEPTION 'Model kind is incompatible with business evidence source'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$
    """)
    for table in ("episodio_atencao", "evento_acao", "sinal_reacao"):
        op.execute(
            f"CREATE TRIGGER business_model BEFORE INSERT OR UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION validate_business_model()"
        )


def _drop_guards() -> None:
    for table, trigger in (
        ("versao_modelo", "immutable_model_version"),
        ("gondola", "geometry_cutover"),
        ("observacao_frame", "observation_camera"),
        ("observacao_interacao", "observation_link"),
        ("episodio_atencao", "business_model"),
        ("evento_acao", "business_model"),
        ("sinal_reacao", "business_model"),
    ):
        op.execute(f"DROP TRIGGER {trigger} ON {table}")
    for function in (
        "reject_model_mutation",
        "validate_geometry_cutover",
        "validate_observation_camera",
        "validate_observation_link",
        "validate_business_model",
    ):
        op.execute(f"DROP FUNCTION {function}()")
