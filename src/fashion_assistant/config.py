"""Central configuration.

Every tunable number in the project lives here, so you can change the model,
catalog size or ranking weights in one place. Values can be overridden with
environment variables (handy in Colab: os.environ["GEMINI_MODEL"] = "...").
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # ---------- paths ----------
    project_root: Path = field(default_factory=lambda: Path(_env("FA_ROOT", Path(__file__).resolve().parents[2])))

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def knowledge_dir(self) -> Path:
        return self.project_root / "knowledge"

    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    # ---------- dataset ----------
    kaggle_competition: str = "h-and-m-personalized-fashion-recommendations"
    # H&M product groups we treat as "accessories". Add "Shoes" if you want footwear too.
    accessory_groups: tuple = ("Accessories", "Bags")
    garment_groups: tuple = ("Garment Upper body", "Garment Lower body", "Garment Full body")
    catalog_size: int = int(_env("FA_CATALOG_SIZE", "3000"))
    trend_recent_weeks: int = 4        # "recent" window for trend velocity
    trend_baseline_weeks: int = 12     # window the recent window is compared against
    eval_window_weeks: int = 8         # baskets used to build the evaluation set

    # ---------- models ----------
    # Google recommends Gemini 3.x for new projects (2.5 models are being retired).
    llm_model: str = _env("GEMINI_MODEL", "gemini-3.8-flash")
    judge_model: str = _env("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite")
    llm_temperature: float = 0.4
    clip_model: str = _env("FA_CLIP_MODEL", "clip-ViT-B-32")             # image + text, same space
    text_embed_model: str = _env("FA_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")  # style knowledge base

    # ---------- agent ----------
    max_agent_steps: int = 4           # max LLM <-> tool round-trips before forcing the ranker
    candidates_per_search: int = 12
    final_top_k: int = 5

    # ---------- influencer score weights (default "editor" persona) ----------
    w_style: float = 0.45
    w_trend: float = 0.30
    w_color: float = 0.25
    tiktok_weight: float = 0.30        # share of trend score taken from TikTok data when loaded

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.image_dir, self.index_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
