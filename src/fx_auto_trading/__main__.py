"""CLI エントリーポイント.

Usage:
    python -m fx_auto_trading collect --from 2020-01-01
    python -m fx_auto_trading collect --update
    python -m fx_auto_trading evaluate
    python -m fx_auto_trading evaluate --horizon 8
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fx-auto-trading: USD/JPY デイトレード予測"
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    collect_p = sub.add_parser("collect", help="データ収集")
    collect_p.add_argument("--from", dest="start_date", type=str)
    collect_p.add_argument("--to", dest="end_date", type=str, default=None)
    collect_p.add_argument("--update", action="store_true")

    # evaluate
    eval_p = sub.add_parser("evaluate", help="検証 + 評価")
    eval_p.add_argument("--horizon", type=int, default=4, help="予測ホライゾン (1/4/8)")
    eval_p.add_argument("--top-features", type=int, default=15, help="使用する特徴量数")

    args = parser.parse_args()

    if args.command == "collect":
        from scripts.collect_data import main as collect_main

        sys.argv = ["collect_data"]
        if args.start_date:
            sys.argv.extend(["--from", args.start_date])
        if args.end_date:
            sys.argv.extend(["--to", args.end_date])
        if args.update:
            sys.argv.append("--update")
        collect_main()

    elif args.command == "evaluate":
        from fx_auto_trading.pipeline import run_evaluate

        run_evaluate(horizon=args.horizon, top_n_features=args.top_features)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
