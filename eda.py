"""EDA: saves visualizations of the revenue data to reports/."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from data_ingestion import ingest_data


def run_eda(data_dir=None, out_dir="reports"):
    data = ingest_data(data_dir or config.get_paths()["data_dir"])
    os.makedirs(out_dir, exist_ok=True)
    countries = data[data.country != "all"].groupby("country").revenue.sum().sort_values()
    allr = data[data.country == "all"].set_index("date")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(allr.revenue.rolling(7).mean(), color="#2E75B6")
    ax[0, 0].set_title("Daily revenue, 7-day average (all countries)")
    ax[0, 1].barh(countries.index, countries.values, color="#1F3A5F")
    ax[0, 1].set_title("Total revenue by country")
    monthly = allr.revenue.resample("MS").sum()
    ax[1, 0].bar(monthly.index, monthly.values, width=20, color="#2E75B6")
    ax[1, 0].set_title("Monthly revenue (all countries)")
    dow = allr.groupby(allr.index.dayofweek).revenue.mean()
    ax[1, 1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.reindex(range(7)).values, color="#C55A11")
    ax[1, 1].set_title("Average revenue by weekday")
    plt.tight_layout()
    path = os.path.join(out_dir, "eda.png")
    plt.savefig(path, dpi=130); plt.close()
    return path


if __name__ == "__main__":
    print(run_eda(sys.argv[1] if len(sys.argv) > 1 else None))
