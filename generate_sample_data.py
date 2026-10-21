"""Creates stand-in invoice files in the same layout as the course data
(one JSON file per month, one record per invoice line).  Replace data/cs-train
with the real course files for the final run."""
import argparse, json, os
import numpy as np
import pandas as pd

COUNTRIES = {"United Kingdom": 60, "France": 8, "Germany": 9, "EIRE": 7, "Spain": 4,
             "Netherlands": 5, "Norway": 3, "Portugal": 3, "Singapore": 2, "Hong Kong": 2}


def generate(out_dir, start="2017-11-01", end="2019-07-31", scale=1.0, seed=42):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    days = pd.date_range(start, end)
    invoice = 100000
    for (y, m), month_days in pd.Series(days, index=days).groupby([days.year, days.month]):
        records = []
        for d in month_days:
            trend = 1 + 0.4 * (d - days[0]).days / len(days)
            season = 1 + 0.25 * np.sin(2 * np.pi * d.dayofyear / 365) + (0.3 if d.month in (11, 12) else 0)
            weekday = 0.5 if d.dayofweek == 6 else 1.0
            for country, base in COUNTRIES.items():
                n = rng.poisson(max(0.2, base * scale * trend * season * weekday / 6))
                for _ in range(n):
                    invoice += 1
                    for _ in range(rng.integers(1, 4)):
                        records.append({"country": country, "customer_id": int(rng.integers(1000, 9000)),
                                        "day": int(d.day), "invoice": str(invoice), "month": int(d.month),
                                        "year": int(d.year), "price": round(float(rng.lognormal(2.0, 0.6)), 2),
                                        "stream_id": str(rng.integers(100, 999)),
                                        "times_viewed": int(rng.integers(1, 12))})
        with open(os.path.join(out_dir, f"invoices-{y}-{m:02d}.json"), "w") as f:
            json.dump(records, f)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "cs-train"))
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    print("wrote", generate(a.out, scale=a.scale))
