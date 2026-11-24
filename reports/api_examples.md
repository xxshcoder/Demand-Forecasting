# Captured API responses

Captured from the locally running service with the production alias active.

## GET /health
```json
{
  "status": "ok",
  "model_loaded": true
}
```
## GET /model-info
```json
{
  "model_name": "demand-forecaster",
  "model_version": "1",
  "alias": "production",
  "training_date": "2026-11-24T15:40:10.482000+00:00",
  "run_id": "3f5a277e7cec44a0b281baef212b2adf",
  "validation_metrics": {
    "mae": 2.1200175826231664,
    "rmse": 3.0333942542170225,
    "mape": 12.555603867580531,
    "wape": 11.969655946516218
  },
  "feature_list": [
    "store_id",
    "item_id",
    "dow",
    "month",
    "day_of_month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "roll_mean_7",
    "roll_std_7",
    "roll_mean_14",
    "roll_std_14",
    "roll_mean_28",
    "roll_std_28"
  ],
  "data_version": "536fde5487f76afc",
  "loaded_at": "2026-11-24T16:05:04.078798+00:00"
}
```
## POST /predict
```json
{
  "prediction": 7.36,
  "model_name": "demand-forecaster",
  "model_version": "1",
  "request_id": "4d0a686a-eefc-4045-a440-dc1737ab21d2",
  "timestamp": "2026-11-24T16:05:05.737156+00:00"
}
```
## Invalid input (27 recent_sales values)
HTTP 422
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": [
        "body",
        "recent_sales"
      ],
      "msg": "Value error, recent_sales must contain exactly 28 values",
      "input": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ],
      "ctx": {
        "error": {}
      }
    }
  ]
}
```
## POST /admin/reload
HTTP 200
```json
{
  "reloaded": true,
  "info": {
    "model_name": "demand-forecaster",
    "model_version": "1",
    "alias": "production",
    "training_date": "2026-11-24T15:40:10.482000+00:00",
    "run_id": "3f5a277e7cec44a0b281baef212b2adf",
    "validation_metrics": {
      "mae": 2.1200175826231664,
      "rmse": 3.0333942542170225,
      "mape": 12.555603867580531,
      "wape": 11.969655946516218
    },
    "feature_list": [
      "store_id",
      "item_id",
      "dow",
      "month",
      "day_of_month",
      "is_weekend",
      "is_holiday",
      "lag_1",
      "lag_7",
      "lag_14",
      "lag_28",
      "roll_mean_7",
      "roll_std_7",
      "roll_mean_14",
      "roll_std_14",
      "roll_mean_28",
      "roll_std_28"
    ],
    "data_version": "536fde5487f76afc",
    "loaded_at": "2026-11-24T16:05:04.078798+00:00"
  },
  "timestamp": "2026-11-24T16:05:05.816183Z"
}
```
