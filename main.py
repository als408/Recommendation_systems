"""Train and evaluate SASRec on the prepared train/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from recommendationsystems.sasrec import fit_sasrec, precision_recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("df_train.csv"))
    parser.add_argument("--test", type=Path, default=Path("df_test.csv"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--filter-seen", choices=("both", "yes", "no"), default="both",
        help="Evaluate with seen-item filtering, without it, or both (default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    recommender, _ = fit_sasrec(
        train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    users = set(train["user"]).intersection(test["user"])
    modes = [True, False] if args.filter_seen == "both" else [args.filter_seen == "yes"]
    for filter_seen in modes:
        precision, recall, count = precision_recall_at_k(
            recommender, test, users=users, k=args.k, filter_seen=filter_seen
        )
        label = "filtered" if filter_seen else "unfiltered"
        print(f"\n--- SASRec / {label} ({count} users) ---")
        print(f"Precision@{args.k}: {precision:.4f}")
        print(f"Recall@{args.k}:    {recall:.4f}")


if __name__ == "__main__":
    main()
