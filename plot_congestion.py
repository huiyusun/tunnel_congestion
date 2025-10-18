#!/usr/bin/env python3
"""
Plot and analyze Lincoln Tunnel congestion data from
data/lincoln_tunnel_crossing_times.csv

Usage Examples:
    # Latest day line plot
    python plot_congestion.py

    # Specific date
    python plot_congestion.py --date 2025-10-15

    # Averages all days within a date range (inclusive)
    python plot_congestion.py --start 2025-10-13 --end 2025-10-20

    # Specific week aggregate
    python plot_congestion.py --aggregate --week 2025-10-14

    # Different Days of the week averaged over all weeks (median minutes by time of day)
    python plot_congestion.py --aggregate

    Plots to add:
    -compare weekends vs weekdays, holidays vs non-holidays
    -plot each Monday, etc. across multiple weeks
    -compare between different months


Flags Summary:
  --date        YYYY-MM-DD for a specific day's line plot.
  --start       Start date (YYYY-MM-DD) to filter the dataset (inclusive).
  --end         End date (YYYY-MM-DD), inclusive.
  --aggregate   Plots median minutes by time of day, split by weekday.
"""

import os
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import matplotlib.pyplot as plt

# ---- Hour-axis helpers ----
_HOUR_LABELS = ["12am", "1am", "2am", "3am", "4am", "5am", "6am", "7am", "8am", "9am", "10am", "11am",
                "12pm", "1pm", "2pm", "3pm", "4pm", "5pm", "6pm", "7pm", "8pm", "9pm", "10pm", "11pm"]

# Gap threshold (minutes) to break lines in daily plots
GAP_MINUTES = 20


def _time_series_to_minutes(hhmm_series: pd.Series) -> pd.Series:
    """Convert 'HH:MM' strings to minutes since midnight (0..1439)."""
    hh = hhmm_series.str.slice(0, 2).astype(int)
    mm = hhmm_series.str.slice(3, 5).astype(int)
    return hh * 60 + mm


def _set_hour_ticks():
    # Major ticks: every hour, label only 12am and 12pm fully
    hour_ticks = list(range(0, 24 * 60, 60))
    hour_labels = []
    for i in range(24):
        if i == 0:
            hour_labels.append("12am")
        elif i == 12:
            hour_labels.append("12pm")
        elif i == 23:
            hour_labels.append("11pm")
        else:
            label = str(i if i <= 12 else i - 12)
            if i < 12:
                label += "am"
            else:
                label += "pm"
            hour_labels.append(label)
    plt.xticks(hour_ticks, hour_labels, rotation=0)

    ax = plt.gca()

    # Minor ticks every 30 minutes (indents + grid)
    ax.set_xticks(range(0, 24 * 60, 30), minor=True)
    ax.tick_params(axis='x', which='minor', length=4, width=0.8)

    # Extend full day range
    plt.xlim(0, 24 * 60 - 1)


# ---- Y-axis/grid helpers ----
def _set_y_ticks_from5_every3(series_list):
    """Set y-ticks every 3 minutes starting at 5, top based on data."""
    import math
    y_max = 0.0
    for s in series_list:
        if s is None:
            continue
        try:
            m = pd.to_numeric(s, errors='coerce').max()
        except Exception:
            m = pd.Series(s).max()
        if pd.notna(m):
            y_max = max(y_max, float(m))
    top = max(5.0, y_max)
    top = int(math.ceil(top / 3.0) * 3)  # round up to next multiple of 3
    # Add slight padding of 3 minutes above top for clarity
    top += 3
    ticks = list(range(5, top + 1, 3))
    if ticks:
        plt.yticks(ticks)


def _apply_grid():
    """Enable semi-visible background grid for both axes, with x-grid on hourly and half-hour marks."""
    ax = plt.gca()
    # x-grid: show at both hourly (major) and 30-min (minor) ticks
    ax.grid(True, which='major', axis='x', alpha=0.3, linestyle='-', linewidth=0.6)
    ax.grid(True, which='minor', axis='x', alpha=0.15, linestyle='-', linewidth=0.4)
    # y-grid: all ticks
    ax.grid(True, which='both', axis='y', alpha=0.25, linestyle='-', linewidth=0.5)


def _auto_legend_loc(x_vals, y_series_list):
    """
    Choose a legend corner with the fewest points using a simple 2x2 bin density heuristic.
    Accepts either a single x series (applied to all y series) or a list of x series
    aligned one-to-one with y_series_list. Robust to differing lengths by aligning by position.
    """
    import numpy as _np

    def _prep_xy(x_in, y_in):
        xv = pd.Series(x_in).reset_index(drop=True)
        yv = pd.to_numeric(pd.Series(y_in), errors='coerce').reset_index(drop=True)
        n = min(len(xv), len(yv))
        if n == 0:
            return None, None
        xv = xv.iloc[:n]
        yv = yv.iloc[:n]
        m = yv.notna().to_numpy()
        return xv.to_numpy()[m], yv.to_numpy()[m]

    xs_all, ys_all = [], []
    if isinstance(x_vals, (list, tuple)):
        for x_in, y_in in zip(x_vals, y_series_list):
            X, Y = _prep_xy(x_in, y_in)
            if X is not None:
                xs_all.append(X)
                ys_all.append(Y)
    else:
        for y_in in y_series_list:
            X, Y = _prep_xy(x_vals, y_in)
            if X is not None:
                xs_all.append(X)
                ys_all.append(Y)

    if not xs_all:
        return 'upper right'

    X = _np.concatenate(xs_all)
    Y = _np.concatenate(ys_all)
    if X.size == 0:
        return 'upper right'

    x0, x1 = X.min(), X.max()
    y0, y1 = Y.min(), Y.max()
    if x1 == x0:
        x1 = x0 + 1.0
    if y1 == y0:
        y1 = y0 + 1.0
    Xn = (X - x0) / (x1 - x0)
    Yn = (Y - y0) / (y1 - y0)

    UL = ((Xn < 0.5) & (Yn >= 0.5)).sum()
    UR = ((Xn >= 0.5) & (Yn >= 0.5)).sum()
    LL = ((Xn < 0.5) & (Yn < 0.5)).sum()
    LR = ((Xn >= 0.5) & (Yn < 0.5)).sum()
    counts = {"upper left": UL, "upper right": UR, "lower left": LL, "lower right": LR}
    return min(counts, key=counts.get)


def _normalize_freq(freq: str) -> str:
    """
    Normalize pandas offset alias: map 'T' to 'min' variants to silence deprecation warnings.
    Examples: '15T' -> '15min', '5T' -> '5min'.
    """
    if isinstance(freq, str) and freq.endswith("T") and freq[:-1].isdigit():
        return f"{freq[:-1]}min"
    return freq


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(BASE_DIR, "data", "lincoln_tunnel_crossing_times.csv")
OUT_DIR = os.path.join(BASE_DIR, "charts")


def apply_date_filter(df, start=None, end=None):
    """
    Filter the dataframe to inclusive [start, end] date range if provided.
    Dates should be strings 'YYYY-MM-DD' or None. Operates on df['date'].
    """
    if start:
        try:
            s = datetime.strptime(start, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"Invalid --start: {start}. Use YYYY-MM-DD.")
        df = df[df["date"] >= s]
    if end:
        try:
            e = datetime.strptime(end, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"Invalid --end: {end}. Use YYYY-MM-DD.")
        df = df[df["date"] <= e]
    if df.empty:
        raise SystemExit("No rows after applying date range filter.")
    return df


def load_df():
    df = pd.read_csv(IN_CSV)
    # timestamps are NY-local strings like "YYYY-MM-DDTHH:MM"
    # parse as timezone-aware in America/New_York
    ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M", errors="coerce")
    ts = ts.dt.tz_localize(ZoneInfo("America/New_York"), nonexistent="NaT", ambiguous="NaT")
    df["timestamp"] = ts
    # coerce numeric and invalidate 0/negative values
    for col in ["time_to_ny", "time_to_nj"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # count how many were invalid or non-positive
        invalid_mask = df[col].isna() | (df[col] <= 0)
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            print(f"[clean] {col}: marking {invalid_count} values as invalid (NaN) due to non-numeric/<=0")
        df.loc[df[col] <= 0, col] = pd.NA
    # add helpers
    df["date"] = df["timestamp"].dt.date
    df["time_of_day"] = df["timestamp"].dt.strftime("%H:%M")
    # drop rows with no timestamp or both minutes missing
    df = df.dropna(subset=["timestamp"])
    df = df[~(df["time_to_ny"].isna() & df["time_to_nj"].isna())]
    # brief summary of retained rows
    print(f"[clean] rows retained for plotting: {len(df)}")
    return df.sort_values("timestamp")


def enrich_time_bins(df, freq="15min"):
    """
    Add weekday name and a time-of-day bin (e.g., 15-minute bins) for heatmaps.
    """
    # Ensure timestamp is present
    if "timestamp" not in df:
        raise ValueError("DataFrame missing 'timestamp' column.")
    out = df.copy()
    out["weekday"] = out["timestamp"].dt.day_name()
    # floor to freq bins for stability (e.g., group 21:01..21:14 into 21:00)
    norm = _normalize_freq(freq)
    floored = out["timestamp"].dt.floor(norm)
    out["time_bin"] = floored.dt.strftime("%H:%M")
    return out


def pick_date(df, date_str=None):
    if date_str:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"Invalid --date: {date_str}. Use YYYY-MM-DD.")
    else:
        # most recent date in data
        d = df["date"].max()
    if pd.isna(d):
        raise SystemExit("No valid timestamps found in CSV.")
    day_df = df[df["date"] == d].copy()
    if day_df.empty:
        raise SystemExit(f"No rows for date {d}.")
    return d, day_df


def plot_day(day_df, day):
    """
    Plot time of day vs minutes for a single day, or for multiple days if day is a list.
    If a list of days/dataframes is provided, overlays all on one plot.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    # If day is a list, average all days and plot the mean congestion per time_of_day
    if isinstance(day, list) and isinstance(day_df, list):
        plt.figure(figsize=(11, 6))
        # Combine all days into one DataFrame
        combined = pd.concat(day_df, ignore_index=True)
        combined = combined.dropna(subset=["timestamp"])
        combined = combined.sort_values("timestamp")
        combined["time_of_day"] = combined["timestamp"].dt.floor("5min").dt.strftime("%H:%M")

        # Compute mean congestion time for each time_of_day
        avg_df = combined.groupby("time_of_day", as_index=False)[["time_to_ny", "time_to_nj"]].mean(numeric_only=True)

        # Prepare x-axis numeric values
        x_num = (pd.to_datetime(avg_df["time_of_day"], format="%H:%M", errors="coerce").dt.hour * 60 +
                 pd.to_datetime(avg_df["time_of_day"], format="%H:%M", errors="coerce").dt.minute).astype(int)

        # Plot averaged lines for NY and NJ
        plt.plot(x_num, avg_df["time_to_ny"], label="Avg To NY (min)", linewidth=1.5, color="C0", alpha=0.9)
        plt.plot(x_num, avg_df["time_to_nj"], label="Avg To NJ (min)", linewidth=1.5, color="C1", alpha=0.9)

        _set_hour_ticks()
        _set_y_ticks_from5_every3([avg_df["time_to_ny"], avg_df["time_to_nj"]])
        _apply_grid()
        plt.legend(loc="upper left", frameon=True, framealpha=0.6)
        plt.title(f"Lincoln Tunnel Congestion — Average over Days ({day[0]} to {day[-1]})", fontsize=14, pad=12)
        plt.xlabel("Time of Day (America/New_York)")
        plt.ylabel("Average Minutes")
        out_png = os.path.join(OUT_DIR, f"time_of_day_avg_{day[0]}_to_{day[-1]}.png")
        plt.tight_layout(pad=1.3)
        plt.savefig(out_png, dpi=160)
        print(f"Saved averaged range plot: {out_png}")
        return

    # Default: plot a single day
    # Only require timestamp; allow one direction to be missing
    day_df = day_df.dropna(subset=["timestamp"])
    day_df = day_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    assert day_df["timestamp"].is_monotonic_increasing, "Timestamps not strictly increasing!"

    plt.figure(figsize=(11, 6))
    x_num = (day_df["timestamp"].dt.hour * 60 + day_df["timestamp"].dt.minute).astype(int)
    gaps = day_df["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    ny_line = day_df["time_to_ny"].astype("float").copy() if "time_to_ny" in day_df else None
    nj_line = day_df["time_to_nj"].astype("float").copy() if "time_to_nj" in day_df else None

    # mark gaps
    num_gaps = int((gaps > GAP_MINUTES).sum())
    if num_gaps:
        print(f"[gap] {num_gaps} time gaps > {GAP_MINUTES} minutes found for {day}")

    # Helper to plot segmented lines where consecutive points are within GAP_MINUTES
    # and both time gaps and missing/invalid y-values cause segment splits.
    def _plot_segmented(x_series, y_series, label, color=None):
        xv = x_series.reset_index(drop=True)
        yv = y_series.reset_index(drop=True).astype(float)
        n = len(xv)
        if n == 0:
            return

        # Build segments of consecutive VALID points where time jumps are <= GAP_MINUTES.
        valid = yv.notna()
        segments = []  # list of (start_idx_inclusive, end_idx_exclusive)
        s = None
        prev_i = None
        for i in range(n):
            if not valid.iloc[i]:
                if s is not None:
                    segments.append((s, prev_i + 1))
                    s = None
                prev_i = i
                continue
            if s is None:
                s = i
            else:
                if (xv.iloc[i] - xv.iloc[prev_i]) > GAP_MINUTES:
                    segments.append((s, prev_i + 1))
                    s = i
            prev_i = i
        if s is not None:
            segments.append((s, prev_i + 1))

        # Plot solid segments
        for k, (s, e) in enumerate(segments):
            if e - s >= 2:
                plt.plot(
                    xv.iloc[s:e],
                    yv.iloc[s:e],
                    label=label if k == 0 else "_nolegend_",
                    linewidth=1.2,
                    color=color,
                )

        # Dotted connectors over any break (time gap or missing values)
        for k in range(len(segments) - 1):
            prev_end_idx = segments[k][1] - 1
            next_start_idx = segments[k + 1][0]
            y_prev = yv.iloc[prev_end_idx]
            y_next = yv.iloc[next_start_idx]
            if pd.notna(y_prev) and pd.notna(y_next):
                plt.plot(
                    [xv.iloc[prev_end_idx], xv.iloc[next_start_idx]],
                    [y_prev, y_next],
                    linestyle=":",
                    linewidth=1.0,
                    color=color,
                    label="_nolegend_",
                )

        # overlay points
        plt.scatter(xv, yv, s=4, alpha=1.0, label="_nolegend_", color=color)

    if ny_line is not None:
        _plot_segmented(x_num, day_df["time_to_ny"], label="To NY (min)", color="C0")
    if nj_line is not None:
        _plot_segmented(x_num, day_df["time_to_nj"], label="To NJ (min)", color="C1")

    _set_hour_ticks()
    _set_y_ticks_from5_every3([day_df["time_to_ny"], day_df["time_to_nj"]])
    _apply_grid()

    plt.title(f"Lincoln Tunnel Congestion — Single day ({day})", fontsize=14, pad=12)
    plt.xlabel("Time of Day (America/New_York)")
    plt.ylabel("Minutes")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="C0", lw=1.2, label="To NY (min)"),
        Line2D([0], [0], color="C1", lw=1.2, label="To NJ (min)"),
    ]
    plt.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)

    out_png = os.path.join(OUT_DIR, f"time_of_day_{day}.png")
    plt.savefig(out_png, dpi=160)
    print(f"Saved {out_png}")


def plot_aggregate(df, week_range=None):
    """
    Median minutes by HH:MM across each weekday (Monday–Sunday).
    Creates one plot with a separate line per weekday for To NY and To NJ.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Add weekday names
    df["weekday"] = df["timestamp"].dt.day_name()

    # mask non-positive values so medians ignore them
    df = df.copy()
    df["time_to_ny"] = df["time_to_ny"].where(df["time_to_ny"] > 0)
    df["time_to_nj"] = df["time_to_nj"].where(df["time_to_nj"] > 0)

    # Aggregate median times per weekday per time_of_day
    agg = df.groupby(["weekday", "time_of_day"], observed=False)[["time_to_ny", "time_to_nj"]].median(numeric_only=True).reset_index()

    # Define weekday order for consistent plotting
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg["weekday"] = pd.Categorical(agg["weekday"], categories=weekday_order, ordered=True)
    agg = agg.sort_values(["weekday", "time_of_day"])

    # Fixed colors per weekday for consistent segment coloring
    weekday_colors = {
        "Monday": "C0",
        "Tuesday": "C1",
        "Wednesday": "C2",
        "Thursday": "C3",
        "Friday": "C4",
        "Saturday": "C5",
        "Sunday": "C6",
    }

    # Plot To NY
    plt.figure(figsize=(12, 7))
    for day in weekday_order:
        day_df = agg[agg["weekday"] == day].copy()
        day_df["x_num"] = (day_df["timestamp"].dt.hour * 60 + day_df["timestamp"].dt.minute).astype(int) if "timestamp" in day_df else None
        if "timestamp" not in day_df or day_df["x_num"] is None:
            times = pd.to_datetime(day_df["time_of_day"], format="%H:%M", errors="coerce")
            day_df["x_num"] = (times.dt.hour * 60 + times.dt.minute).astype(int)
        # Only require x_num; allow one direction median to be missing for that bin
        day_df = day_df.dropna(subset=["x_num"])
        day_df = day_df.sort_values("x_num").drop_duplicates(subset=["x_num"]).reset_index(drop=True)
        assert day_df["x_num"].is_monotonic_increasing, f"x_num not strictly increasing for {day}!"
        gaps = day_df["x_num"].diff().fillna(0)
        num_gaps = int((gaps > GAP_MINUTES).sum())
        if num_gaps:
            print(f"[gap] {num_gaps} time gaps > {GAP_MINUTES} minutes found for {day}")

        # Plot segmented lines (split on time gaps or invalid y)
        xv = day_df["x_num"].reset_index(drop=True)
        yv = day_df["time_to_ny"].reset_index(drop=True).astype(float)
        n = len(xv)
        valid = yv.notna()
        segments = []
        s_idx = None
        prev_i = None
        for i in range(n):
            if not valid.iloc[i]:
                if s_idx is not None:
                    segments.append((s_idx, prev_i + 1))
                    s_idx = None
                prev_i = i
                continue
            if s_idx is None:
                s_idx = i
            else:
                if (xv.iloc[i] - xv.iloc[prev_i]) > GAP_MINUTES:
                    segments.append((s_idx, prev_i + 1))
                    s_idx = i
            prev_i = i
        if s_idx is not None:
            segments.append((s_idx, prev_i + 1))

        color = weekday_colors.get(day)
        for k, (s_idx, e_idx) in enumerate(segments):
            if e_idx - s_idx >= 2:
                plt.plot(
                    xv.iloc[s_idx:e_idx], yv.iloc[s_idx:e_idx],
                    label=day if k == 0 else "_nolegend_",
                    linewidth=1.2,
                    color=color,
                )
        # dotted connectors across any break
        for k in range(len(segments) - 1):
            prev_end_idx = segments[k][1] - 1
            next_start_idx = segments[k + 1][0]
            y_prev = yv.iloc[prev_end_idx]
            y_next = yv.iloc[next_start_idx]
            if pd.notna(y_prev) and pd.notna(y_next):
                plt.plot(
                    [xv.iloc[prev_end_idx], xv.iloc[next_start_idx]],
                    [y_prev, y_next],
                    linestyle=":",
                    linewidth=1.0,
                    color=color,
                    label="_nolegend_",
                )
        plt.scatter(xv, yv, s=4, alpha=1.0, label="_nolegend_", color=weekday_colors.get(day))
    if week_range:
        plt.title(f"Lincoln Tunnel Congestion — Aggregate by days of the week (to NY) ({week_range[0]} to {week_range[1]})", fontsize=14, pad=12)
    else:
        plt.title("Lincoln Tunnel Congestion — Aggregate by days of the week (to NY)", fontsize=14, pad=12)
    plt.xlabel("Time of Day (America/New_York)")
    plt.ylabel("Median Minutes (To NY)")
    _set_hour_ticks()
    _set_y_ticks_from5_every3([agg["time_to_ny"]])
    _apply_grid()

    from matplotlib.lines import Line2D
    present_days = [d for d in weekday_order if not agg[agg["weekday"] == d].empty]
    handles = [Line2D([0], [0], color=weekday_colors[d], lw=1.2, label=d) for d in present_days]

    plt.legend(handles=handles, title="Weekday", ncol=2, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)

    out_png_ny = os.path.join(OUT_DIR, "time_of_day_by_weekday_to_ny.png")
    plt.savefig(out_png_ny, dpi=160)
    print(f"Saved {out_png_ny}")

    # Plot To NJ
    plt.figure(figsize=(12, 7))
    for day in weekday_order:
        day_df = agg[agg["weekday"] == day].copy()
        day_df["x_num"] = (day_df["timestamp"].dt.hour * 60 + day_df["timestamp"].dt.minute).astype(int) if "timestamp" in day_df else None
        if "timestamp" not in day_df or day_df["x_num"] is None:
            times = pd.to_datetime(day_df["time_of_day"], format="%H:%M", errors="coerce")
            day_df["x_num"] = (times.dt.hour * 60 + times.dt.minute).astype(int)
        # Only require x_num; allow one direction median to be missing for that bin
        day_df = day_df.dropna(subset=["x_num"])
        day_df = day_df.sort_values("x_num").drop_duplicates(subset=["x_num"]).reset_index(drop=True)
        assert day_df["x_num"].is_monotonic_increasing, f"x_num not strictly increasing for {day}!"
        gaps = day_df["x_num"].diff().fillna(0)
        num_gaps = int((gaps > GAP_MINUTES).sum())
        if num_gaps:
            print(f"[gap] {num_gaps} time gaps > {GAP_MINUTES} minutes found for {day}")

        # Plot segmented lines (split on time gaps or invalid y)
        xv = day_df["x_num"].reset_index(drop=True)
        yv = day_df["time_to_nj"].reset_index(drop=True).astype(float)
        n = len(xv)
        valid = yv.notna()
        segments = []
        s_idx = None
        prev_i = None
        for i in range(n):
            if not valid.iloc[i]:
                if s_idx is not None:
                    segments.append((s_idx, prev_i + 1))
                    s_idx = None
                prev_i = i
                continue
            if s_idx is None:
                s_idx = i
            else:
                if (xv.iloc[i] - xv.iloc[prev_i]) > GAP_MINUTES:
                    segments.append((s_idx, prev_i + 1))
                    s_idx = i
            prev_i = i
        if s_idx is not None:
            segments.append((s_idx, prev_i + 1))

        color = weekday_colors.get(day)
        for k, (s_idx, e_idx) in enumerate(segments):
            if e_idx - s_idx >= 2:
                plt.plot(
                    xv.iloc[s_idx:e_idx], yv.iloc[s_idx:e_idx],
                    label=day if k == 0 else "_nolegend_",
                    linewidth=1.2,
                    color=color,
                )
        # dotted connectors across any break
        for k in range(len(segments) - 1):
            prev_end_idx = segments[k][1] - 1
            next_start_idx = segments[k + 1][0]
            y_prev = yv.iloc[prev_end_idx]
            y_next = yv.iloc[next_start_idx]
            if pd.notna(y_prev) and pd.notna(y_next):
                plt.plot(
                    [xv.iloc[prev_end_idx], xv.iloc[next_start_idx]],
                    [y_prev, y_next],
                    linestyle=":",
                    linewidth=1.0,
                    color=color,
                    label="_nolegend_",
                )
        plt.scatter(xv, yv, s=4, alpha=1.0, label="_nolegend_", color=weekday_colors.get(day))
    if week_range:
        plt.title(f"Lincoln Tunnel Congestion — Aggregate by days of the week (to NJ) ({week_range[0]} to {week_range[1]})", fontsize=14, pad=12)
    else:
        plt.title("Lincoln Tunnel Congestion — Aggregate by days of the week (to NJ)", fontsize=14, pad=12)
    plt.xlabel("Time of Day (America/New_York)")
    plt.ylabel("Median Minutes (To NJ)")
    _set_hour_ticks()
    _set_y_ticks_from5_every3([agg["time_to_nj"]])
    _apply_grid()

    from matplotlib.lines import Line2D
    present_days = [d for d in weekday_order if not agg[agg["weekday"] == d].empty]
    handles = [Line2D([0], [0], color=weekday_colors[d], lw=1.2, label=d) for d in present_days]

    plt.legend(handles=handles, title="Weekday", ncol=2, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)

    out_png_nj = os.path.join(OUT_DIR, "time_of_day_by_weekday_to_nj.png")
    plt.savefig(out_png_nj, dpi=160)
    print(f"Saved {out_png_nj}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (defaults to most recent date in data)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--aggregate", action="store_true", help="also produce weekday median plots (Mon–Sun)")
    parser.add_argument("--week", help="Any date within the week (YYYY-MM-DD) to limit aggregation to that week")
    args = parser.parse_args()

    df = load_df()
    # Apply date filters if provided (affects all downstream plots)
    if args.start or args.end:
        df = apply_date_filter(df, start=args.start, end=args.end)

    # Plot logic for day/range
    if args.date:
        day, day_df = pick_date(df, args.date)
        plot_day(day_df, day)
    elif args.start or args.end:
        unique_days = sorted(df["date"].unique())
        # plot all days in range on one combined graph
        day_dfs = []
        for day in unique_days:
            day_df = df[df["date"] == day]
            day_dfs.append(day_df)
        plot_day(day_dfs, unique_days)
    else:
        day, day_df = pick_date(df, None)
        plot_day(day_df, day)

    if args.aggregate:
        if args.week:
            try:
                week_date = datetime.strptime(args.week, "%Y-%m-%d").date()
                week_start = week_date - timedelta(days=week_date.weekday())  # Monday
                week_end = week_start + timedelta(days=6)  # Sunday
                df_week = apply_date_filter(df, start=week_start.isoformat(), end=week_end.isoformat())
                plot_aggregate(df_week, week_range=(week_start, week_end))
            except ValueError:
                raise SystemExit(f"Invalid --week: {args.week}. Use YYYY-MM-DD.")
        else:
            plot_aggregate(df)


if __name__ == "__main__":
    main()
