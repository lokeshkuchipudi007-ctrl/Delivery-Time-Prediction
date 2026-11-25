# Delivery Time Prediction

A small Flask web app that predicts food/parcel delivery time (in minutes) from
distance, preparation time, courier experience, traffic and weather.

The model is a scikit-learn `GradientBoostingRegressor` pipeline trained on
synthetic data. Swap `generate_data()` in `train_model.py` for your real dataset
to get real predictions.

## Project structure

```
Delivery_Time_Prediction/
├── app.py            # Flask app (web form + JSON API)
├── train_model.py    # Trains the model and writes model.pkl
├── model.pkl         # Trained model (generated)
├── requirements.txt
├── Dockerfile
├── templates/index.html
├── static/style.css
└── tests/test_app.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_model.py            # creates model.pkl
python app.py                    # http://localhost:5000
```

## API

`POST /api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "distance_km": 5,
        "prep_time_min": 15,
        "courier_experience_years": 3,
        "traffic": "medium",
        "weather": "clear"
      }'
```

Response: `{"delivery_time_min": 34.2}`

| Field | Type | Allowed values |
|---|---|---|
| `distance_km` | number | 0.1 – 100 |
| `prep_time_min` | number | 0 – 120 |
| `courier_experience_years` | number | 0 – 40 |
| `traffic` | string | `low`, `medium`, `high` |
| `weather` | string | `clear`, `rain`, `storm` |

Invalid input returns HTTP 400 with `{"error": "..."}`.
`GET /health` returns service status.

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t delivery-time-prediction .
docker run -p 5000:5000 delivery-time-prediction
```

The image trains the model at build time if `model.pkl` is not present.
