"""Discord Webhook通知 (TASK-002-02)."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def send_signal(
    direction: str,
    price: float,
    probability: float,
    adx: float,
    atr: float,
    sl_price: float,
    tp_price: float,
    timestamp: str,
) -> bool:
    """取引シグナルをDiscordに通知する."""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL が未設定")
        return False

    if direction == "buy":
        emoji = "🟢"
        dir_text = "買い"
    elif direction == "sell":
        emoji = "🔴"
        dir_text = "売り"
    else:
        return False

    content = (
        f"{emoji} **{dir_text} USD/JPY @ {price:.2f}**\n"
        f"　確率: {probability:.2f} | ADX: {adx:.1f} | ATR: {atr:.3f}\n"
        f"　損切り: {sl_price:.2f} | 利確: {tp_price:.2f}\n"
        f"　時刻: {timestamp}"
    )

    try:
        resp = httpx.post(url, json={"content": content}, timeout=10)
        if resp.status_code == 204:
            logger.info("Discord通知送信完了")
            return True
        logger.warning("Discord通知失敗: %d", resp.status_code)
        return False
    except httpx.HTTPError as e:
        logger.warning("Discord通知エラー: %s", e)
        return False


def send_error(message: str) -> bool:
    """エラーをDiscordに通知する."""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        return False

    content = f"⚠️ **FX Auto Trading エラー**\n{message}"

    try:
        resp = httpx.post(url, json={"content": content}, timeout=10)
        return resp.status_code == 204
    except httpx.HTTPError:
        return False
