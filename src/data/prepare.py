"""CLI entry point for the DVC ``prepare`` stage: raw CSV -> parquet."""

from __future__ import annotations

import argparse

from ..config import load_params, resolve_path
from .clean import clean_sales
from .load import load_raw
from .validate import validate_processed, validate_raw


def prepare(raw_path: str | None = None, out_path: str | None = None, params: dict | None = None):
    """Load, validate, clean, re-validate and persist the processed panel."""
    params = params or load_params()
    raw_path = raw_path or params["data"]["raw_path"]
    out_path = out_path or params["data"]["processed_path"]

    raw = load_raw(resolve_path(raw_path))
    validate_raw(raw)
    processed = clean_sales(raw, params)
    validate_processed(processed)

    destination = resolve_path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(destination, index=False, engine="pyarrow")
    print(f"Wrote {len(processed):,} rows to {destination}")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", dest="raw_path", default=None)
    parser.add_argument("--out", dest="out_path", default=None)
    args = parser.parse_args()
    prepare(args.raw_path, args.out_path)


if __name__ == "__main__":
    main()
