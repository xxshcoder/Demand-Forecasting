"""Measure real API latency with concurrent requests against a running service."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PAYLOAD = {
    "store_id": 1,
    "item_id": 5,
    "forecast_date": "2018-01-01",
    "recent_sales": [12.0] * 28,
    "price": 9.99,
    "promo": 0,
}


async def measure(url: str, total: int, concurrency: int) -> dict:
    latencies: list[float] = []
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=url, timeout=30.0) as client:

        async def one_request() -> None:
            nonlocal errors
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post("/predict", json=PAYLOAD)
                    if response.status_code != 200:
                        errors += 1
                except httpx.HTTPError:
                    errors += 1
                latencies.append((time.perf_counter() - started) * 1000)

        started_all = time.perf_counter()
        await asyncio.gather(*(one_request() for _ in range(total)))
        elapsed = time.perf_counter() - started_all

    if not latencies:
        raise RuntimeError("No requests completed; is the API running?")
    ordered = sorted(latencies)
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "url": url,
        "requests": total,
        "concurrency": concurrency,
        "errors": errors,
        "error_rate": errors / total,
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "mean_ms": statistics.fmean(ordered),
        "throughput_rps": total / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--out", default="reports/latency.json", type=Path)
    args = parser.parse_args()
    result = asyncio.run(measure(args.url, args.requests, args.concurrency))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
