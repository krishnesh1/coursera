"""Model training, comparison with a baseline, saving/loading and prediction.

Task: forecast revenue for the next 30 days from the history up to a given date,
for one country or for all countries combined."""
import os, time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

import config
from data_ingestion import ingest_data, normalise_country
from logger import update_predict_log, update_train_log

HORIZON, MIN_HISTORY = 30, 90
MODEL_VERSION = "1.0"


def features_for(revenue, i):
    """Features using only data up to and including day i (revenue is a numpy array)."""
    def s(n):
        return revenue[max(0, i - n + 1):i + 1].sum()
    return [s(7), s(14), s(30), s(60), s(90)]


def build_dataset(series):
    """series: DataFrame with date, revenue for ONE country, sorted by date."""
    rev = series.revenue.to_numpy(dtype=float)
    dates = series.date.to_numpy()
    X, y, d, base = [], [], [], []
    for i in range(MIN_HISTORY - 1, len(rev) - HORIZON):
        month = pd.Timestamp(dates[i]).month
        X.append(features_for(rev, i) + [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)])
        y.append(rev[i + 1:i + 1 + HORIZON].sum())
        base.append(rev[max(0, i - HORIZON + 1):i + 1].sum())   # baseline: last 30 days repeat
        d.append(dates[i])
    return np.array(X), np.array(y), np.array(d), np.array(base)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def candidate_models():
    return {"ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(random_state=42)}


def train_country(series, country, model_dir, log_dir, make_plot_dir=None):
    start = time.time()
    X, y, dates, base = build_dataset(series)
    if len(y) < 60:
        raise ValueError(f"not enough history to train {country}")
    cut = int(len(y) * 0.8)
    gap = HORIZON                        # no overlap between train targets and test period
    Xtr, ytr = X[:cut - gap], y[:cut - gap]
    Xte, yte, bte = X[cut:], y[cut:], base[cut:]
    scores, fitted = {}, {}
    for name, m in candidate_models().items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        scores[name] = rmse(yte, pred)
        fitted[name] = m
    baseline_rmse = rmse(yte, bte)
    best = min(scores, key=scores.get)
    final = candidate_models()[best].fit(X, y)       # refit best model on all data
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"model": final, "name": best, "version": MODEL_VERSION,
                 "train_mean": float(y.mean()), "train_std": float(y.std()),
                 "rmse": scores[best], "baseline_rmse": baseline_rmse},
                os.path.join(model_dir, f"model-{country}.joblib"))
    update_train_log(log_dir, country, best, scores[best], baseline_rmse, len(y),
                     time.time() - start, MODEL_VERSION)
    if make_plot_dir:
        plot_comparison(country, dates[cut:], yte, bte, fitted[best].predict(Xte), best,
                        scores[best], baseline_rmse, make_plot_dir)
    return {"country": country, "best_model": best, "rmse": scores[best],
            "baseline_rmse": baseline_rmse, "all_scores": scores}


def plot_comparison(country, dates, actual, baseline, predicted, name, m_rmse, b_rmse, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].plot(dates, actual, label="Actual", color="black")
    ax[0].plot(dates, baseline, label="Baseline (last 30 days repeated)", color="#999999", ls="--")
    ax[0].plot(dates, predicted, label=f"Model ({name})", color="#2E75B6")
    ax[0].set_title(f"Next-30-day revenue, test period ({country})"); ax[0].legend()
    ax[0].set_ylabel("Revenue")
    ax[1].bar(["Baseline", f"Model"], [b_rmse, m_rmse], color=["#999999", "#2E75B6"])
    ax[1].set_title("Test RMSE (lower is better)")
    plt.tight_layout()
    path = os.path.join(out_dir, f"model_vs_baseline_{country}.png")
    plt.savefig(path, dpi=130); plt.close()
    return path


def model_train(data_dir=None, test=False, countries=None, top_n=10, plot_dir=None):
    """Train one model per country (top_n by revenue) plus 'all'. test=True uses
    the test model/log folders so production files are never touched."""
    paths = config.get_paths(test)
    data = ingest_data(data_dir or paths["data_dir"], top_n=top_n)
    wanted = countries or sorted(data.country.unique())
    results = []
    for c in wanted:
        s = data[data.country == c].sort_values("date")
        results.append(train_country(s, c, paths["model_dir"], paths["log_dir"],
                                     make_plot_dir=plot_dir if c == "all" else None))
    return results


def model_load(country, test=False):
    path = os.path.join(config.get_paths(test)["model_dir"], f"model-{normalise_country(country)}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained model for '{country}'")
    return joblib.load(path)


def model_predict(country, date, test=False, data_dir=None):
    """Forecast revenue for the 30 days after `date` using history up to `date`."""
    start = time.time()
    paths = config.get_paths(test)
    country = normalise_country(country)
    bundle = model_load(country, test)
    data = ingest_data(data_dir or paths["data_dir"])
    s = data[data.country == country].sort_values("date").reset_index(drop=True)
    if s.empty:
        raise ValueError(f"unknown country '{country}'")
    ts = pd.Timestamp(date)
    hist = s[s.date <= ts]
    if len(hist) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} days of history before {date}")
    rev = hist.revenue.to_numpy(dtype=float)
    i = len(rev) - 1
    x = features_for(rev, i) + [np.sin(2 * np.pi * ts.month / 12), np.cos(2 * np.pi * ts.month / 12)]
    pred = float(max(0.0, bundle["model"].predict(np.array([x]))[0]))
    update_predict_log(paths["log_dir"], country, str(ts.date()), pred, bundle["version"],
                       time.time() - start)
    return {"country": country, "date": str(ts.date()), "predicted_revenue_next_30_days": round(pred, 2),
            "model": bundle["name"], "model_version": bundle["version"]}
