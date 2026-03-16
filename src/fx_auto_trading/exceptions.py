"""カスタム例外クラス."""

from __future__ import annotations


class FxAutoTradingError(Exception):
    """基底例外."""


class ApiError(FxAutoTradingError):
    """GMO FX API 通信エラー."""


class ApiRateLimitError(ApiError):
    """レート制限超過."""


class ApiResponseError(ApiError):
    """API レスポンスエラー."""


class StorageError(FxAutoTradingError):
    """SQLite ストレージエラー."""


class DataQualityError(FxAutoTradingError):
    """データ品質チェックエラー."""


class ModelError(FxAutoTradingError):
    """モデル訓練・予測エラー."""


class OverfitWarning(FxAutoTradingError):
    """過学習警告 (ADR 004 #2: 精度差>10%)."""
