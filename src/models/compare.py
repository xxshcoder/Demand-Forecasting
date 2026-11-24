"""Write the honest model comparison table and interpretation."""

from __future__ import annotations

import pandas as pd

from ..config import ROOT, load_params
from .evaluate import evaluate_saved_predictions

MODEL_ORDER = ["last_week", "moving_average", "linear", "hgb", "nn"]
LEARNED = ["linear", "hgb", "nn"]


def build_comparison(params: dict | None = None) -> pd.DataFrame:
    params = params or load_params()
    evaluation = evaluate_saved_predictions(params)
    metrics = evaluation["metrics"]
    rows = []
    for name in MODEL_ORDER:
        values = metrics[name]
        rows.append(
            {
                "model": name,
                "MAE": values["mae"],
                "RMSE": values["rmse"],
                "MAPE": values["mape"],
                "WAPE": values["wape"],
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    primary = str(params["promotion"]["primary_metric"]).lower()
    best = min(LEARNED, key=lambda name: table.loc[name, primary.upper()])
    baseline = table.loc["moving_average", primary.upper()]
    best_value = table.loc[best, primary.upper()]
    improvement = (baseline - best_value) / baseline * 100 if baseline else float("nan")
    if best_value >= baseline:
        interpretation = (
            f"No learned model beat the moving-average baseline on {primary.upper()} "
            f"({best_value:.3f} vs {baseline:.3f}). This is a negative result, reported as-is."
        )
    else:
        interpretation = (
            f"Best learned model: **{best}**, with {primary.upper()} {best_value:.3f} versus "
            f"{baseline:.3f} for moving average ({improvement:.1f}% lower). "
        )
        nn_mae, hgb_mae = table.loc["nn", "MAE"], table.loc["hgb", "MAE"]
        relation = "beat" if nn_mae < hgb_mae else "did not beat"
        interpretation += (
            f"The neural net {relation} gradient boosting on MAE "
            f"({nn_mae:.3f} vs {hgb_mae:.3f}). All models were scored on the same "
            f"{evaluation['validation_rows']:,} validation rows."
        )
    # Keep the numeric table numeric; the narrative is metadata so pandas 3
    # does not have to downcast a float column to object.
    table.attrs["interpretation"] = interpretation
    return table


def main() -> None:
    params = load_params()
    table = build_comparison(params)
    interpretation = str(table.attrs["interpretation"])
    table.to_csv(ROOT / "reports" / "model_comparison.csv")
    rounded = table.round(3)
    header = "| model | MAE | RMSE | MAPE | WAPE |\n|---|---:|---:|---:|---:|"
    body = "\n".join(
        f"| {name} | {row['MAE']:.3f} | {row['RMSE']:.3f} | {row['MAPE']:.3f} | {row['WAPE']:.3f} |"
        for name, row in rounded.iterrows()
    )
    markdown = f"{header}\n{body}\n\n{interpretation}\n"
    (ROOT / "reports" / "model_comparison.md").write_text(markdown, encoding="utf-8")
    # Do not rewrite evaluation.json: it is an output of the evaluate stage, and
    # mutating it here makes DVC see a permanently dirty pipeline.
    print(markdown)


if __name__ == "__main__":
    main()
