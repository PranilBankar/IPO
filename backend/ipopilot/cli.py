"""IPOPilot command line.

    python -m ipopilot.cli bootstrap          # first run: universe, market, live, backfills
    python -m ipopilot.cli sync               # refresh live issues + GMP
    python -m ipopilot.cli backfill-subs      # final category-wise subscription history
    python -m ipopilot.cli backfill-gmp       # IPOWatch day-wise GMP history
    python -m ipopilot.cli news [--history]   # headlines + FinBERT scores
    python -m ipopilot.cli train              # train + evaluate listing-return models
    python -m ipopilot.cli backtest           # strategy comparison (research question)
    python -m ipopilot.cli serve              # FastAPI on :8000
"""
from __future__ import annotations

import argparse
import json
import logging


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ipopilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    b = sub.add_parser("bootstrap")
    b.add_argument("--quick", action="store_true", help="skip the long historical backfills")
    sub.add_parser("sync")
    sub.add_parser("index")
    sub.add_parser("market")
    bs = sub.add_parser("backfill-subs")
    bs.add_argument("--limit", type=int)
    bg = sub.add_parser("backfill-gmp")
    bg.add_argument("--since", type=int, default=2019)
    bg.add_argument("--limit", type=int)
    nw = sub.add_parser("news")
    nw.add_argument("--history", action="store_true", help="also fetch headlines for listed IPOs since 2023")
    nw.add_argument("--limit", type=int, default=400)
    tr = sub.add_parser("train")
    tr.add_argument("--epochs", type=int, default=200)
    tr.add_argument("--ensemble", type=int, default=5)
    sub.add_parser("backtest")
    sub.add_parser("predict")
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .db import init_db
    init_db()
    from .ingestion import pipeline as pl

    out = None
    if a.cmd == "init":
        pl.ensure_demo_ipo()
        out = "ok"
    elif a.cmd == "bootstrap":
        out = pl.bootstrap(full=not a.quick)
    elif a.cmd == "sync":
        out = pl.run_all_live()
    elif a.cmd == "index":
        out = pl.sync_index()
    elif a.cmd == "market":
        out = pl.sync_market()
    elif a.cmd == "backfill-subs":
        out = pl.backfill_subscriptions(limit=a.limit)
    elif a.cmd == "backfill-gmp":
        out = pl.backfill_gmp(since_year=a.since, limit=a.limit)
    elif a.cmd == "news":
        slugs = None
        if a.history:
            from sqlalchemy import select
            from .db import Ipo, session_scope
            with session_scope() as s:
                slugs = [i.slug for i in s.scalars(select(Ipo).where(
                    Ipo.listing_gain_pct.is_not(None), Ipo.close_date >= "2023-01-01")
                    .order_by(Ipo.close_date.desc()).limit(a.limit))]
        out = pl.sync_news(slugs)
    elif a.cmd == "train":
        from .ml.train import train_all
        out = train_all(epochs=a.epochs, ensemble=a.ensemble)
    elif a.cmd == "backtest":
        from .engine.backtest import run_backtest
        out = run_backtest()
    elif a.cmd == "predict":
        from .engine.service import refresh_predictions
        out = refresh_predictions()
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("ipopilot.api.app:app", host=a.host, port=a.port, reload=False)
        return
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
