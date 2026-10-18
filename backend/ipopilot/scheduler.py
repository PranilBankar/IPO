"""Background refresh jobs (APScheduler, in-process).

  * every N minutes on weekdays 09:00-18:00 IST : live issues, subscription, GMP
  * hourly                                       : news headlines + FinBERT scores
  * daily 19:30 IST                              : IPO index (listing gains), market bars,
                                                    prediction snapshot, forecaster refit
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import settings

log = logging.getLogger(__name__)
_sched: BackgroundScheduler | None = None


def _live():
    from .engine.service import registry
    from .ingestion import pipeline
    pipeline.sync_live()
    pipeline.sync_gmp_live()
    registry.invalidate()


def _news():
    from .ingestion import pipeline
    pipeline.sync_news()


def _daily():
    from .engine.service import refresh_predictions
    from .engine.subscription import fit_from_history
    from .ingestion import pipeline
    pipeline.sync_index()
    pipeline.sync_market()
    fit_from_history()
    refresh_predictions()


def start_scheduler() -> BackgroundScheduler:
    global _sched
    if _sched is not None:
        return _sched
    tz = "Asia/Kolkata"
    s = BackgroundScheduler(timezone=tz, job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 600})
    s.add_job(_live, CronTrigger(day_of_week="mon-fri", hour="9-18", minute=f"*/{settings.refresh_minutes}", timezone=tz),
              id="live")
    s.add_job(_news, CronTrigger(minute=15, timezone=tz), id="news")
    s.add_job(_daily, CronTrigger(hour=19, minute=30, timezone=tz), id="daily")
    s.start()
    _sched = s
    log.info("scheduler started (live every %d min)", settings.refresh_minutes)
    return s
