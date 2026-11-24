| model | MAE | RMSE | MAPE | WAPE |
|---|---:|---:|---:|---:|
| last_week | 3.471 | 5.075 | 19.528 | 19.596 |
| moving_average | 3.351 | 4.745 | 19.629 | 18.922 |
| linear | 2.588 | 3.742 | 15.252 | 14.612 |
| hgb | 2.120 | 3.033 | 12.556 | 11.970 |
| nn | 2.257 | 3.247 | 13.194 | 12.741 |

Best learned model: **hgb**, with MAE 2.120 versus 3.351 for moving average (36.7% lower). The neural net did not beat gradient boosting on MAE (2.257 vs 2.120). All models were scored on the same 45,000 validation rows.
