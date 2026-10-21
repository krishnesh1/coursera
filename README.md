# AI Workflow Capstone: revenue forecasting API

Forecasts the next 30 days of revenue for one country or for all countries combined.

## Files
| File | Purpose |
|---|---|
| `data_ingestion.py` | `ingest_data(data_dir)`: loads the monthly JSON invoice files into a daily series per country plus `all` |
| `eda.py` | EDA visualizations saved to `reports/eda.png` |
| `model.py` | Trains ridge, random forest and gradient boosting, compares them with a baseline (last 30 days repeated), saves the best, predicts; saves `reports/model_vs_baseline_all.png` |
| `app.py` | Flask API: `/train`, `/predict`, `/monitor`, `/health` |
| `logger.py` | Train and predict logs |
| `monitoring.py` | Performance report: RMSE vs baseline, prediction latency, drift alert |
| `unittests/` | Tests for the API, the model and the logging |
| `run_tests.py` | Runs all unit tests with one command |
| `Dockerfile` | Container image |

## Use your real data
Put the course JSON files (e.g. `invoices-2017-11.json`) in `data/cs-train/`.
For a quick test without them: `python generate_sample_data.py` (stand-in data only).

## Run locally
    pip install -r requirements.txt
    python eda.py
    python -c "import model; model.model_train(plot_dir='reports')"
    python app.py
    curl -X POST localhost:8080/predict -H 'Content-Type: application/json' -d '{"country":"all","date":"2019-06-15"}'
    curl "localhost:8080/predict?country=united_kingdom&date=2019-06-15"
    curl localhost:8080/monitor

## Tests
    python run_tests.py
Tests set `CAPSTONE_HOME` to a temp folder and use `test=True`, so they never touch `models/` or `logs/`.

## Docker
    docker build -t capstone .
    docker run -p 8080:8080 capstone
With real data: `docker run -p 8080:8080 -v $(pwd)/data/cs-train:/app/data/cs-train capstone`,
then call `POST /train` once so the models are trained on it.
