"""Performance monitoring built from the train and predict logs.

Reports: model quality vs baseline (from training), prediction counts and
latency, and a simple drift alert when recent predictions move far away from
what the model saw in training."""
import os
import numpy as np
import pandas as pd
import config
from model import model_load

LATENCY_ALERT_SECONDS = 5.0
DRIFT_Z_ALERT = 3.0


def _read(log_dir, name):
    path = os.path.join(log_dir, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def performance_report(test=False, last_n=50):
    log_dir = config.get_paths(test)["log_dir"]
    train, pred = _read(log_dir, "train.log"), _read(log_dir, "predict.log")
    report = {"models": {}, "predictions": {}, "alerts": []}
    if not train.empty:
        latest = train.sort_values("timestamp").groupby("country").tail(1)
        for _, r in latest.iterrows():
            gain = 1 - r.rmse / r.baseline_rmse if r.baseline_rmse else 0.0
            report["models"][r.country] = {"model": r.model, "rmse": r.rmse,
                                           "baseline_rmse": r.baseline_rmse,
                                           "improvement_over_baseline": round(float(gain), 4)}
            if r.rmse >= r.baseline_rmse:
                report["alerts"].append(f"{r.country}: model is not better than the baseline")
    if not pred.empty:
        recent = pred.sort_values("timestamp").tail(last_n)
        report["predictions"] = {"count": int(len(pred)),
                                 "mean_runtime_seconds": round(float(recent.runtime_seconds.mean()), 4),
                                 "max_runtime_seconds": round(float(recent.runtime_seconds.max()), 4)}
        if recent.runtime_seconds.mean() > LATENCY_ALERT_SECONDS:
            report["alerts"].append("mean prediction runtime above threshold")
        for country, g in recent.groupby("country"):
            try:
                b = model_load(country, test)
            except FileNotFoundError:
                continue
            if b["train_std"] > 0:
                z = abs(g.predicted_revenue.mean() - b["train_mean"]) / b["train_std"]
                if z > DRIFT_Z_ALERT:
                    report["alerts"].append(f"{country}: predictions drifted from training range (z={z:.1f})")
    return report
