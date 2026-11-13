"""Database schema (SQLAlchemy 2.0). SQLite by default; set DATABASE_URL for PostgreSQL.

Design rule: observations (subscription, GMP, news, market bars) are APPEND-ONLY and
time-stamped, so that every model can be trained on exactly what was knowable at the
time of a decision (point-in-time correctness, no look-ahead leakage).
"""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import (
    JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
    create_engine, event, select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import settings


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Ipo(Base):
    """One public issue. `slug` (from ipos.markets) is the canonical cross-source key."""

    __tablename__ = "ipos"

    slug: Mapped[str] = mapped_column(String(160), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    symbol: Mapped[str | None] = mapped_column(String(40), index=True)
    bse_code: Mapped[str | None] = mapped_column(String(20))
    board: Mapped[str] = mapped_column(String(16), index=True)  # mainboard | sme
    offer_type: Mapped[str | None] = mapped_column(String(16))
    stage: Mapped[str | None] = mapped_column(String(24), index=True)  # upcoming/open/closed/listed/...
    open_date: Mapped[dt.date | None] = mapped_column(Date, index=True)
    close_date: Mapped[dt.date | None] = mapped_column(Date, index=True)
    listing_date: Mapped[dt.date | None] = mapped_column(Date)
    price_low: Mapped[float | None] = mapped_column(Float)
    price_high: Mapped[float | None] = mapped_column(Float)
    issue_price: Mapped[float | None] = mapped_column(Float)  # final price (after pricing)
    lot_size: Mapped[int | None] = mapped_column(Integer)
    issue_size_cr: Mapped[float | None] = mapped_column(Float)
    sector: Mapped[str | None] = mapped_column(String(80))
    anchor_pct: Mapped[float | None] = mapped_column(Float)
    partly_paid: Mapped[bool] = mapped_column(Boolean, default=False)
    listing_gain_pct: Mapped[float | None] = mapped_column(Float)  # training label
    cmp: Mapped[float | None] = mapped_column(Float)
    registrar: Mapped[str | None] = mapped_column(String(120))
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)  # fictional demo issue
    extra: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    subscriptions: Mapped[list["SubscriptionSnapshot"]] = relationship(back_populates="ipo")
    gmps: Mapped[list["GmpObservation"]] = relationship(back_populates="ipo")

    @property
    def price(self) -> float | None:
        return self.issue_price or self.price_high or self.price_low


class SubscriptionSnapshot(Base):
    __tablename__ = "subscription_snapshots"
    __table_args__ = (UniqueConstraint("ipo_slug", "source", "as_of", name="uq_sub_obs"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ipo_slug: Mapped[str] = mapped_column(ForeignKey("ipos.slug"), index=True)
    source: Mapped[str] = mapped_column(String(40))
    as_of: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    final: Mapped[bool] = mapped_column(Boolean, default=False)
    total_times: Mapped[float | None] = mapped_column(Float)
    qib_times: Mapped[float | None] = mapped_column(Float)
    nii_times: Mapped[float | None] = mapped_column(Float)
    snii_times: Mapped[float | None] = mapped_column(Float)  # NII 2-10 lakh
    bnii_times: Mapped[float | None] = mapped_column(Float)  # NII >10 lakh
    rii_times: Mapped[float | None] = mapped_column(Float)
    emp_times: Mapped[float | None] = mapped_column(Float)
    rii_offered: Mapped[float | None] = mapped_column(Float)
    rii_bid: Mapped[float | None] = mapped_column(Float)
    snii_offered: Mapped[float | None] = mapped_column(Float)
    snii_bid: Mapped[float | None] = mapped_column(Float)
    rii_applications: Mapped[float | None] = mapped_column(Float)
    raw: Mapped[dict | None] = mapped_column(JSON)

    ipo: Mapped[Ipo] = relationship(back_populates="subscriptions")


class GmpObservation(Base):
    __tablename__ = "gmp_observations"
    __table_args__ = (UniqueConstraint("ipo_slug", "source", "obs_date", name="uq_gmp_obs"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ipo_slug: Mapped[str] = mapped_column(ForeignKey("ipos.slug"), index=True)
    source: Mapped[str] = mapped_column(String(40))
    obs_date: Mapped[dt.date] = mapped_column(Date, index=True)  # the day the GMP refers to
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    gmp_rs: Mapped[float | None] = mapped_column(Float)
    gmp_pct: Mapped[float | None] = mapped_column(Float)
    ref_price: Mapped[float | None] = mapped_column(Float)
    raw: Mapped[dict | None] = mapped_column(JSON)

    ipo: Mapped[Ipo] = relationship(back_populates="gmps")


class NewsItem(Base):
    __tablename__ = "news_items"
    __table_args__ = (UniqueConstraint("ipo_slug", "url", name="uq_news"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ipo_slug: Mapped[str] = mapped_column(ForeignKey("ipos.slug"), index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str] = mapped_column(String(1000))
    publisher: Mapped[str | None] = mapped_column(String(200))
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    p_pos: Mapped[float | None] = mapped_column(Float)
    p_neg: Mapped[float | None] = mapped_column(Float)
    p_neu: Mapped[float | None] = mapped_column(Float)
    topic: Mapped[str | None] = mapped_column(String(40))


class MarketBar(Base):
    __tablename__ = "market_bars"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_bar"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    close: Mapped[float] = mapped_column(Float)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ipo_slug: Mapped[str] = mapped_column(ForeignKey("ipos.slug"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(60))
    mean_pct: Mapped[float] = mapped_column(Float)
    median_pct: Mapped[float] = mapped_column(Float)
    q10_pct: Mapped[float] = mapped_column(Float)
    q90_pct: Mapped[float] = mapped_column(Float)
    p_loss: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict | None] = mapped_column(JSON)


class PlanRecord(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    request: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict] = mapped_column(JSON)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job: Mapped[str] = mapped_column(String(60))
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    ok: Mapped[bool] = mapped_column(Boolean, default=False)
    stats: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)


engine = create_engine(
    settings.db_url,
    future=True,
    connect_args={"check_same_thread": False, "timeout": 30} if settings.db_url.startswith("sqlite") else {},
)

if settings.db_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(conn, _):  # pragma: no cover - trivial
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


__all__ = [
    "Base", "Ipo", "SubscriptionSnapshot", "GmpObservation", "NewsItem", "MarketBar", "Prediction",
    "PlanRecord", "IngestRun", "engine", "SessionLocal", "init_db", "session_scope", "get_session",
    "select", "utcnow",
]
