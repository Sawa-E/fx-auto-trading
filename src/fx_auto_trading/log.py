"""ログ設定."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """アプリケーション全体のログ設定を初期化する."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
