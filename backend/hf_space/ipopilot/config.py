"""Central configuration. Every value can be overridden with an environment variable
(or a `.env` file in the repo root / backend folder)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    for p in (REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("IPOPILOT_DATA_DIR", str(REPO_ROOT / "data"))))
    database_url: str = _env("DATABASE_URL", "")
    user_agent: str = _env(
        "IPOPILOT_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36",
    )
    # polite scraping: minimum seconds between requests to the same host
    min_request_interval: float = float(_env("IPOPILOT_MIN_REQUEST_INTERVAL", "1.0"))
    http_timeout: float = float(_env("IPOPILOT_HTTP_TIMEOUT", "25"))
    cors_origins: str = _env("IPOPILOT_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    sentiment_model: str = _env("IPOPILOT_SENTIMENT_MODEL", "ProsusAI/finbert")
    enable_scheduler: bool = _env("IPOPILOT_ENABLE_SCHEDULER", "1") == "1"
    # minutes between live refreshes while markets are open
    refresh_minutes: int = int(_env("IPOPILOT_REFRESH_MINUTES", "30"))
    # cost assumptions used by the optimizer (per allotted lot, rupees)
    sell_cost_pct: float = float(_env("IPOPILOT_SELL_COST_PCT", "0.25"))  # brokerage+STT+charges on sale, %
    admin_api_key: str = _env("IPOPILOT_ADMIN_KEY", "")

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'ipopilot.db').as_posix()}"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.raw_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
