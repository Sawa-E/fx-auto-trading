"""histdata.comから1分足データをDL → 1H足に集約 → SQLite保存.

Usage: python scripts/download_histdata.py
"""

from __future__ import annotations

import glob
import os
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from fx_auto_trading.data.storage import OhlcStorage
from fx_auto_trading.log import setup_logging

setup_logging()

OUTPUT_DIR = "histdata_tmp"
PAIR = "usdjpy"
SYMBOL = "USD_JPY"
INTERVAL = "1hour"


def download_all() -> None:
    """2000年〜現在までの1分足データをダウンロードする."""
    from histdata import download_hist_data as dl
    from histdata.api import Platform as P, TimeFrame as TF

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    os.chdir(OUTPUT_DIR)

    current_year = datetime.now().year

    # 過去年は年単位
    for year in range(2000, current_year):
        fname = f"DAT_ASCII_{PAIR.upper()}_M1_{year}.zip"
        if os.path.exists(fname):
            print(f"  スキップ（既存）: {year}")
            continue
        print(f"  ダウンロード: {year}...")
        try:
            dl(
                year=str(year),
                pair=PAIR,
                platform=P.GENERIC_ASCII,
                time_frame=TF.ONE_MINUTE,
            )
        except Exception as e:
            print(f"  エラー: {year} → {e}")

    # 今年は月単位
    for month in range(1, 13):
        fname = f"DAT_ASCII_{PAIR.upper()}_M1_{current_year}{month:02d}.zip"
        if os.path.exists(fname):
            print(f"  スキップ（既存）: {current_year}/{month:02d}")
            continue
        print(f"  ダウンロード: {current_year}/{month:02d}...")
        try:
            dl(
                year=str(current_year),
                month=str(month),
                pair=PAIR,
                platform=P.GENERIC_ASCII,
                time_frame=TF.ONE_MINUTE,
            )
        except Exception as e:
            print(f"  エラー: {current_year}/{month:02d} → {e}")
            break  # 未来月はエラーになるので終了

    os.chdir("..")


def extract_and_resample() -> pd.DataFrame:
    """ZIPを展開し、1分足→1H足にリサンプリングする."""
    all_dfs: list[pd.DataFrame] = []

    for zf in sorted(glob.glob(f"{OUTPUT_DIR}/DAT_ASCII_{PAIR.upper()}_M1_*.zip")):
        print(f"  処理中: {Path(zf).name}")
        with zipfile.ZipFile(zf) as z:
            for name in z.namelist():
                if not name.endswith(".csv"):
                    continue
                with z.open(name) as f:
                    df = pd.read_csv(
                        f,
                        sep=";",
                        names=["datetime", "open", "high", "low", "close", "volume"],
                        header=None,
                    )
                    # 末尾に空列がある場合を除去
                    df = df.dropna(axis=1, how="all")
                    df = df[["datetime", "open", "high", "low", "close"]]
                    df["datetime"] = pd.to_datetime(
                        df["datetime"].str.strip(), format="%Y%m%d %H%M%S"
                    )
                    df = df.set_index("datetime")
                    df.index = df.index.tz_localize("US/Eastern").tz_convert("UTC")
                    all_dfs.append(df)

    if not all_dfs:
        print("データなし")
        return pd.DataFrame()

    print("  結合中...")
    full = pd.concat(all_dfs).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    print(f"  1分足: {len(full)}本 → 1H足にリサンプリング...")
    df_1h = full.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()

    print(f"  1H足: {len(df_1h)}本")
    return df_1h


def save_to_db(df_1h: pd.DataFrame) -> None:
    """1H足データをSQLiteに保存（GMOデータと結合）."""
    storage = OhlcStorage("data/fx_auto_trading.db")

    # GMOデータの最古timestampを取得
    existing = storage.load(SYMBOL, INTERVAL)
    if not existing.empty:
        gmo_start = existing.index[0]
        # histdataはGMO以前の分のみ保存
        df_1h = df_1h[df_1h.index < gmo_start]
        print(f"  GMOデータ({gmo_start})より前の分のみ保存: {len(df_1h)}本")

    if df_1h.empty:
        print("  保存対象なし（GMOデータと重複）")
        return

    saved = storage.save(df_1h, SYMBOL, INTERVAL)
    total = storage.count(SYMBOL, INTERVAL)
    print(f"  保存完了: {len(df_1h)}本, DB合計: {total}本")


def main() -> None:
    print("=== histdata.com からUSD/JPY 1分足ダウンロード ===")
    download_all()

    print("\n=== 1分足 → 1H足 リサンプリング ===")
    df_1h = extract_and_resample()
    if df_1h.empty:
        return

    print(f"\n期間: {df_1h.index[0]} ~ {df_1h.index[-1]}")
    print(f"合計: {len(df_1h)}本")

    print("\n=== SQLite保存 ===")
    save_to_db(df_1h)

    # DB全体の確認
    storage = OhlcStorage("data/fx_auto_trading.db")
    full = storage.load(SYMBOL, INTERVAL)
    print(f"\n=== DB全体 ===")
    print(f"期間: {full.index[0]} ~ {full.index[-1]}")
    print(f"合計: {len(full)}本")


if __name__ == "__main__":
    main()
