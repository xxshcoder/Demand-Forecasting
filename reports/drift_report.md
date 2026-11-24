# Drift report

Reference rows: 809,000. Recent logged rows: 200.
Drift flagged for: lag_1, lag_7, lag_14, lag_28, roll_mean_7, roll_std_7, roll_mean_14, roll_std_14, roll_mean_28, roll_std_28.

| feature | PSI | KS statistic | KS p-value | drift |
|---|---:|---:|---:|:---:|
| store_id | 0.0279 | 0.0700 | 0.2682 | no |
| item_id | 0.0875 | 0.0500 | 0.6803 | no |
| dow | 0.0272 | 0.0455 | 0.7837 | no |
| month | 0.0293 | 0.0276 | 0.9971 | no |
| day_of_month | 0.0251 | 0.0766 | 0.1818 | no |
| is_weekend | 0.0000 | 0.0055 | 1 | no |
| is_holiday | 0.0000 | 0.0090 | 1 | no |
| lag_1 | 0.4734 | 0.2931 | 1.033e-15 | YES |
| lag_7 | 0.3800 | 0.2826 | 1.272e-14 | YES |
| lag_14 | 0.4545 | 0.2820 | 1.441e-14 | YES |
| lag_28 | 0.4597 | 0.2795 | 2.635e-14 | YES |
| roll_mean_7 | 0.5337 | 0.3277 | 1.257e-19 | YES |
| roll_std_7 | 0.4189 | 0.2833 | 1.068e-14 | YES |
| roll_mean_14 | 0.5533 | 0.3268 | 1.585e-19 | YES |
| roll_std_14 | 0.4187 | 0.2852 | 6.908e-15 | YES |
| roll_mean_28 | 0.5543 | 0.3165 | 2.616e-18 | YES |
| roll_std_28 | 0.4111 | 0.2888 | 2.863e-15 | YES |
