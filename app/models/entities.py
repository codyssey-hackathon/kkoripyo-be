"""꼬리표 핵심 데이터 모델.

분석(analysis) 1회 = 계정·게시물·상호작용 수집 → 단서(clue) 추출 → 가설(hypothesis)과 근거(evidence) 산출.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    subject_handle: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(
        String(16), default="pending"
    )  # pending|running|done|failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    handle_hash: Mapped[str] = mapped_column(String(64), index=True)  # 제3자 계정은 해시로만 저장
    is_subject: Mapped[bool] = mapped_column(Boolean, default=False)
    kind: Mapped[str] = mapped_column(String(16), default="unknown")  # person|institution|unknown
    source: Mapped[str] = mapped_column(String(16))  # api|crawl|export|synthetic


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    shortcode: Mapped[str | None] = mapped_column(String(32))
    taken_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    caption: Mapped[str | None] = mapped_column(Text)
    location_tag: Mapped[str | None] = mapped_column(String(256))
    media: Mapped[list | None] = mapped_column(JSON)


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    src_account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    dst_account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    type: Mapped[str] = mapped_column(String(16))  # comment|mention|tag|follows|followed_by
    post_id: Mapped[int | None] = mapped_column(ForeignKey("posts.id"))
    ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Clue(Base):
    __tablename__ = "clues"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    subject_ref: Mapped[str] = mapped_column(String(32))  # 'post:12' | 'account:7'
    clue_type: Mapped[str] = mapped_column(String(32))  # direct_text|ocr_text|visual_logo|...
    attribute: Mapped[str] = mapped_column(String(32))  # school|age_band|work|area|interest|device
    value_norm: Mapped[str] = mapped_column(String(256))
    raw_span: Mapped[str | None] = mapped_column(Text)
    extractor: Mapped[str] = mapped_column(String(16))  # llm|vlm|rule
    extractor_conf: Mapped[float] = mapped_column(Float, default=0.5)
    origin_key: Mapped[str] = mapped_column(String(128), index=True)  # 독립성 판정 키


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    attribute: Mapped[str] = mapped_column(String(32))
    value_norm: Mapped[str] = mapped_column(String(256))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    specificity: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(
        String(16), default="withheld"
    )  # inferred|possible|withheld


class EvidenceLink(Base):
    __tablename__ = "evidence_links"

    id: Mapped[int] = mapped_column(primary_key=True)
    hypothesis_id: Mapped[int] = mapped_column(ForeignKey("hypotheses.id"), index=True)
    clue_id: Mapped[int] = mapped_column(ForeignKey("clues.id"))
    role: Mapped[str] = mapped_column(String(16), default="support")  # support|contradict
    independence_group: Mapped[str] = mapped_column(String(128))
    weight_used: Mapped[float] = mapped_column(Float, default=0.0)


class AgentStep(Base):
    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict | None] = mapped_column(JSON)
    result_summary: Mapped[str | None] = mapped_column(Text)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
