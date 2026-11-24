"""Seed normal and intentionally shifted traffic so drift can be demonstrated.

Normal requests are sampled from real processed histories (all store/item series
and dates). The shifted batch reuses those same requests but doubles every
recent-sales value, so only the sales-derived features move.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HISTORY = 28


def sample_requests(processed: pd.DataFrame, n: int, seed: int = 42) -> list[dict]:
    """Build ``n`` requests from 28 observed sales ending the day before forecast."""
    rng = np.random.default_rng(seed)
    sorted_frame = processed.sort_values(["store_id", "item_id", "date"])
    groups = [
        (int(store), int(item), group["sales"].to_numpy(dtype=float), group["date"].to_numpy())
        for (store, item), group in sorted_frame.groupby(["store_id", "item_id"], sort=True)
        if len(group) > HISTORY
    ]
    if not groups:
        raise ValueError("Processed data has no series with enough history to sample")

    requests = []
    for _ in range(n):
        store, item, sales, dates = groups[int(rng.integers(len(groups)))]
        end = int(rng.integers(HISTORY, len(sales)))
        window = sales[end - HISTORY : end]
        forecast_date = pd.Timestamp(dates[end]).strftime("%Y-%m-%d")
        requests.append(
            {
                "store_id": store,
                "item_id": item,
                "forecast_date": forecast_date,
                "recent_sales": window.tolist(),
                "price": 9.99,
                "promo": 0,
            }
        )
    return requests


def send(client: httpx.Client, requests: list[dict], shift: float = 1.0) -> int:
    """Post requests, optionally scaling the recent-sales history."""
    ok = 0
    for request in requests:
        payload = dict(request)
        if shift != 1.0:
            payload["recent_sales"] = [value * shift for value in request["recent_sales"]]
        ok += client.post("/predict", json=payload).status_code == 200
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--normal", type=int, default=100)
    parser.add_argument("--shifted", type=int, default=100)
    parser.add_argument("--processed", default="data/processed/sales.parquet", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = pd.read_parquet(args.processed)
    normal = sample_requests(processed, args.normal, args.seed)
    shifted = sample_requests(processed, args.shifted, args.seed + 1)
    with httpx.Client(base_url=args.url, timeout=30.0) as client:
        normal_ok = send(client, normal)
        shifted_ok = send(client, shifted, shift=2.0)
    print(f"Normal: {normal_ok}/{args.normal} OK; shifted: {shifted_ok}/{args.shifted} OK")


if __name__ == "__main__":
    main()
