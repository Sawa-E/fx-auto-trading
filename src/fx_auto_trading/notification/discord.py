"""Discord Webhook通知."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def _send(content: str) -> bool:
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL が未設定")
        return False
    try:
        resp = httpx.post(url, json={"content": content}, timeout=10)
        if resp.status_code == 204:
            return True
        logger.warning("Discord通知失敗: %d", resp.status_code)
        return False
    except httpx.HTTPError as e:
        logger.warning("Discord通知エラー: %s", e)
        return False


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
    """エントリーシグナルを通知."""
    emoji = "🟢" if direction == "buy" else "🔴"
    dir_text = "買い" if direction == "buy" else "売り"
    content = (
        f"{emoji} **シグナル: {dir_text} USD/JPY @ {price:.2f}**\n"
        f"　確率: {probability:.2f} | ADX: {adx:.1f} | ATR: {atr:.3f}\n"
        f"　損切り: {sl_price:.2f} | 利確: {tp_price:.2f}\n"
        f"　時刻: {timestamp}"
    )
    return _send(content)


def send_trade_result(
    direction: str,
    entry_price: float,
    exit_price: float,
    result: str,
    pnl_yen: float,
    balance: float,
    wins: int,
    losses: int,
) -> bool:
    """決済結果を通知."""
    if result == "tp_hit":
        emoji = "💰"
        result_text = "利確ヒット!"
    else:
        emoji = "📉"
        result_text = "損切り"

    dir_text = "買い" if direction == "buy" else "売り"
    content = (
        f"{emoji} **{result_text} {pnl_yen:+,.0f}円**\n"
        f"　{dir_text} {entry_price:.2f} → {exit_price:.2f}\n"
        f"　残高: {balance:,.0f}円 | 通算: {wins}勝{losses}敗"
    )
    return _send(content)


def send_error(message: str) -> bool:
    """エラーを通知."""
    return _send(f"⚠️ **FX Auto Trading エラー**\n{message}")
