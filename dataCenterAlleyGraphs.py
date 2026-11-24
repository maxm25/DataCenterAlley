# =============================================================================
# Data Center Alley Pollutants: Daily Average Graphs
# =============================================================================
#
# Turns the raw air quality sensor CSVs in this folder into one daily average
# graph per pollutant (AQI, NO2, CO, VOCs, PM2.5), each with a trend line.
#
# Data source: Virginia Department of Environmental Quality (DEQ)
#   Sensor data:          https://kunakcloud.com/websites/VirginiaDEQ.html
#   Latest DEQ analysis:  https://www.deq.virginia.gov/home/showpublisheddocument/37962
#
# PROGRAM FLOW
# ------------
#   main()
#    |
#    |-- 1. load_sensor_data()
#    |       Reads all 9 sensor CSVs (one per location) and stacks them into
#    |       one table. Each row is one sensor's reading for one hour.
#    |
#    |-- For each pollutant:
#    |    |
#    |    |-- 2. hourly_averages()
#    |    |       Averages all sensors that reported during the same hour,
#    |    |       giving one value per hour for the whole region.
#    |    |
#    |    |-- 3. daily_averages()
#    |    |       Averages the hourly values within each day,
#    |    |       giving one value per day.
#    |    |
#    |    |-- 4. omit_fire_days()
#    |    |       Removes wildfire smoke days (Jul 17-20) so they don't skew
#    |    |       the trend. By default the first 3 (Jul 17-19) are removed.
#    |    |
#    |    |-- 5. fit_trend_line()
#    |    |       Fits a straight line through the daily values and measures
#    |    |       how well it fits (R²).
#    |    |
#    |    '-- 6. plot_daily_average()
#    |            Draws the daily values and trend line and saves the graph to
#    |            DataCenterAlleyDailyGraphs/<Pollutant>DailyGraph.png
#
# HOW TO RUN
# ----------
#   python dataCenterAlleyGraphs.py                 # remove 3 fire days (default)
#   python dataCenterAlleyGraphs.py --fire-days 0   # keep every day
#   python dataCenterAlleyGraphs.py --fire-days 4   # remove all 4 fire days
#   python dataCenterAlleyGraphs.py --show          # also open each graph in a window
# =============================================================================

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------- Settings --------------------------------------

# Folder the sensor CSVs are read from (the same folder as this script),
# and folder the finished graphs are saved to.
SENSOR_DIR = Path(__file__).resolve().parent
GRAPH_DIR = SENSOR_DIR / "DataCenterAlleyDailyGraphs"

# Pollutants to graph. Each entry is:
#   (column name in the sensor CSVs, name used in graph titles/filenames, unit)
POLLUTANTS = [
    ("AQI",           "AQI",   ""),
    ("NO2 (ppb)",     "NO2",   "(ppb)"),
    ("CO (ppm)",      "CO",    "(ppm)"),
    ("VOCs (ppb)",    "VOCs",  "(ppb)"),
    ("PM2.5 (µg/m³)", "PM2.5", "(µg/m³)"),
]

# Days when wildfire smoke affected the readings. --fire-days N removes the
# first N of these. The published graphs remove 3 (Jul 17-19).
FIRE_DAYS = ["2026-07-17", "2026-07-18", "2026-07-19", "2026-07-20"]
DEFAULT_FIRE_DAYS_OMITTED = 3

# The order the sensor files are stacked in.
# Averages are rounded to 2 decimals, and when an average lands exactly halfway
# (e.g. 11.025) the order the numbers were added in can tip it up or down.
# Keeping this fixed order makes the results match the original analysis.
# Any new sensor CSV is added after these, alphabetically.
SENSOR_ORDER = [
    "Newberry Condo Assoc", "Broad Run HS", "Belfort Park Dr", "Sterling MS",
    "1757 Golf Club", "Farmwell Station MS", "Heritage Farm Museum",
    "Steuart Weller ES", "Dulles Airport",
]


# ----------------------------- Step 1: Load ----------------------------------

# Finds every sensor CSV in the folder and returns them in SENSOR_ORDER.
def find_sensor_files(folder):
    def order(path):
        if path.stem in SENSOR_ORDER:
            return (SENSOR_ORDER.index(path.stem), path.stem)
        return (len(SENSOR_ORDER), path.stem)

    files = sorted(folder.glob("*.csv"), key=order)
    if not files:
        raise FileNotFoundError(f"No sensor CSVs found in {folder}")
    return files


# Reads one sensor CSV into a table.
# The DEQ exports are separated by ";" and start with an extra "sep=;" line,
# which is skipped so the real column names are used as the header.
def read_sensor_file(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        first_line = file.readline()
        if not first_line.startswith("sep="):
            file.seek(0)
        return pd.read_csv(file, sep=";", dtype=str)


# Reads every sensor CSV and stacks them into one table with columns:
#   Location, Date, AQI, NO2 (ppb), CO (ppm), VOCs (ppb), PM2.5 (µg/m³)
# Dates like "Aug 6, 2026, 23:00:00" are converted to real date/times
# so they can be sorted and grouped.
def load_sensor_data(folder=SENSOR_DIR):
    tables = [read_sensor_file(path) for path in find_sensor_files(folder)]
    data = pd.concat(tables, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"].str.strip(), format="%b %d, %Y, %H:%M:%S")
    return data


# ----------------------------- Steps 2-3: Average ----------------------------

# Averages a list of numbers and rounds to 2 decimals.
# Adds the numbers one at a time on purpose: Python's built-in sum() adds
# floats more precisely, which rounds a few halfway values differently from
# the original analysis.
def rounded_mean(values):
    total = 0.0
    for value in values:
        total += value
    return round(total / len(values), 2)


# Step 2: Takes the combined sensor table and one pollutant column, and returns
# one value per hour: the average of every sensor that reported that hour.
# Result is a Series indexed by hour, e.g. 2026-02-24 19:00 -> 6.0
def hourly_averages(data, column):
    readings = data[["Date", column]].copy()
    readings[column] = readings[column].astype(float)
    # Sort by time (pandas' default sort; see SENSOR_ORDER for why order matters)
    readings = readings.sort_values("Date")
    return readings.groupby("Date")[column].agg(lambda hour: rounded_mean(hour.tolist()))


# Step 3: Takes the hourly averages and returns one value per day: the average
# of that day's hourly values.
# Result is a table with columns Date and the pollutant, e.g. 2026-02-24 -> 6.8
def daily_averages(hourly):
    days = hourly.index.normalize()  # 2026-02-24 19:00 -> 2026-02-24
    daily = hourly.groupby(days).agg(lambda day: rounded_mean(day.tolist()))
    daily.index.name = "Date"
    return daily.reset_index()


# ----------------------------- Step 4: Fire days -----------------------------

# Removes the first `days_omitted` fire days (0-4) from the daily table.
def omit_fire_days(daily, days_omitted):
    days_omitted = max(0, min(days_omitted, len(FIRE_DAYS)))
    fire_dates = pd.to_datetime(FIRE_DAYS[:days_omitted])
    return daily[~daily["Date"].isin(fire_dates)]


# ----------------------------- Step 5: Trend line ----------------------------

# Fits a straight line y = slope * x + intercept through the daily values,
# where x counts the days shown on the graph (0, 1, 2, ...).
# Returns the line's y value for each day, the slope, the intercept, and
# R² (how much of the day-to-day variation the line explains, from 0 to 1).
def fit_trend_line(y):
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    trend = slope * x + intercept

    leftover_variation = np.sum((y - trend) ** 2)
    total_variation = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - leftover_variation / total_variation if total_variation != 0 else 1.0
    return trend, slope, intercept, r2


# ----------------------------- Step 6: Graph ---------------------------------

# Draws one pollutant's daily averages (blue dashed line with dots) and its
# trend line (red), labels the trend equation and R², and saves the graph as
# a PNG. Returns the equation text so main() can print it.
def plot_daily_average(daily, column, pollutant, unit, out_path, show=False):
    y = daily[column].values
    trend, slope, intercept, r2 = fit_trend_line(y)
    equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}"

    plt.figure(figsize=(10, 6))
    plt.plot(daily["Date"], y, marker="o", linestyle="--", label=f"{pollutant} Daily Average")
    plt.plot(daily["Date"], trend, color="red", linestyle="-", label="Trend Line")
    plt.title(f"{pollutant} Daily Average")
    plt.xlabel("Date")
    plt.ylabel(f"{pollutant} {unit}")
    plt.legend()
    plt.text(0.02, 0.95, equation, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(rotation=45)

    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()
    return equation


# ----------------------------- Run everything --------------------------------

# Reads the options typed after the script name (--fire-days, --show).
def parse_options():
    parser = argparse.ArgumentParser(
        description="Graph daily average pollutant levels from the Data Center Alley sensors.")
    parser.add_argument("--fire-days", type=int, default=DEFAULT_FIRE_DAYS_OMITTED,
                        help="how many of the fire days Jul 17-20 to remove, 0-4 "
                             f"(default: {DEFAULT_FIRE_DAYS_OMITTED})")
    parser.add_argument("--show", action="store_true",
                        help="also open each graph in a window")
    return parser.parse_args()


# Runs steps 1-6: loads the sensor data once, then for each pollutant averages
# it by hour and day, removes fire days, and saves its graph.
def main():
    options = parse_options()

    print("All data used is from the Virginia Department of Environmental Quality (DEQ) "
          "and is publicly available. The data can be accessed at: "
          "https://kunakcloud.com/websites/VirginiaDEQ.html")
    print("The most recent data analysis of said data can be found here: "
          "https://www.deq.virginia.gov/home/showpublisheddocument/37962")

    data = load_sensor_data()                                  # Step 1
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    for column, pollutant, unit in POLLUTANTS:
        hourly = hourly_averages(data, column)                 # Step 2
        daily = daily_averages(hourly)                         # Step 3
        daily = omit_fire_days(daily, options.fire_days)       # Step 4
        out_path = GRAPH_DIR / f"{pollutant}DailyGraph.png"
        equation = plot_daily_average(daily, column, pollutant, unit,
                                      out_path, options.show)  # Steps 5-6
        print(f"{pollutant}: {equation.replace(chr(10), ', ')} -> {out_path.name}")


if __name__ == "__main__":
    main()
