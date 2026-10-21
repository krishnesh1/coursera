"""Data ingestion: one function that turns the raw monthly JSON files into a
clean daily time series per country (plus an 'all' series)."""
import glob, json, os
import pandas as pd

REQUIRED = {"country", "day", "month", "year", "price", "invoice"}


def normalise_country(name):
    return str(name).strip().lower().replace(" ", "_")


def ingest_data(data_dir, top_n=None):
    """Return a DataFrame: date, country, revenue, purchases, invoices.
    Missing days are filled with zero so every series is continuous.
    top_n keeps only the n countries with most revenue (plus 'all')."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no .json files found in {data_dir}")
    frames = []
    for f in files:
        with open(f) as fh:
            frames.append(pd.DataFrame(json.load(fh)))
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.lower().replace("total_price", "price") for c in raw.columns]
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    raw["date"] = pd.to_datetime(raw[["year", "month", "day"]], errors="coerce")
    raw = raw.dropna(subset=["date", "price"])
    raw["country"] = raw["country"].map(normalise_country)

    daily = (raw.groupby(["country", "date"])
                .agg(revenue=("price", "sum"), purchases=("price", "size"),
                     invoices=("invoice", "nunique")).reset_index())
    if top_n:
        keep = daily.groupby("country")["revenue"].sum().nlargest(top_n).index
        daily = daily[daily.country.isin(keep)]
    all_days = (daily.groupby("date")[["revenue", "purchases", "invoices"]].sum().reset_index())
    all_days["country"] = "all"
    daily = pd.concat([daily, all_days], ignore_index=True)

    full = pd.date_range(daily.date.min(), daily.date.max())
    parts = []
    for c, g in daily.groupby("country"):
        g = g.set_index("date").reindex(full).fillna({"revenue": 0, "purchases": 0, "invoices": 0})
        g["country"] = c
        parts.append(g.rename_axis("date").reset_index())
    return pd.concat(parts, ignore_index=True)[["date", "country", "revenue", "purchases", "invoices"]]
