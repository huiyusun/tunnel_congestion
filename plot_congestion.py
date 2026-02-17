#!/usr/bin/env python3
"""
Plot and analyze Lincoln Tunnel congestion data from
data/lincoln_tunnel_crossing_times.csv

Usage Examples:
    # Single: Latest day; specific date; averages all days within range
    python plot_congestion.py
    python plot_congestion.py --date 2025-10-17
    python plot_congestion.py --start 2025-10-13 --end 2026-11-30

    # Aggregate: all weeks averaged; specific weeks; rush vs non-rush periods
    python plot_congestion.py --aggregate
    python plot_congestion.py --aggregate --week 2026-01-05 --week 2026-11-01
    python plot_congestion.py --aggregate --blocks

    # Compare: weekdays vs weekends; holidays vs non-holidays; each tuesdays with each other
    python plot_congestion.py --aggregate --compare weekday_weekend
    python plot_congestion.py --aggregate --compare holiday
    python plot_congestion.py --compare tuesday

    # Compare: different months; average rush vs non-rush periods; tuesday rush vs non-rush periods;
    python plot_congestion.py --compare months
    python plot_congestion.py --compare time_of_day_blocks
    python plot_congestion.py --compare blocks --weekday tuesday
"""

import os
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as USFederalHolidayCalendar

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
        print(f"[warn] No data found for range {start or '?'} to {end or '?'}, returning empty DataFrame.")
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
        plt.title(f"Lincoln Tunnel — Average over Days ({day[0]} to {day[-1]})", fontsize=14, pad=12)
        plt.xlabel("Time of Day (New York)")
        plt.ylabel("Minutes")
        out_png = os.path.join(OUT_DIR, f"average_{day[0]}_to_{day[-1]}.png")
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

    plt.title(f"Lincoln Tunnel — Single day ({day})", fontsize=14, pad=12)
    plt.xlabel("Time of Day (New York)")
    plt.ylabel("Minutes")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="C0", lw=1.2, label="To NY (min)"),
        Line2D([0], [0], color="C1", lw=1.2, label="To NJ (min)"),
    ]
    plt.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)

    out_png = os.path.join(OUT_DIR, f"{day}.png")
    plt.savefig(out_png, dpi=160)
    print(f"Saved {out_png}")


def plot_aggregate(df, week_range=None, week_dates=None):
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

    # Aggregate mean times per weekday per time_of_day
    agg = df.groupby(["weekday", "time_of_day"], observed=False)[["time_to_ny", "time_to_nj"]].mean(numeric_only=True).reset_index()

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
        # Removed scatter: plt.scatter(xv, yv, s=4, alpha=1.0, label="_nolegend_", color=weekday_colors.get(day))
    # Title logic for To NY
    if week_dates:
        if len(week_dates) == 1:
            plt.title(f"Lincoln Tunnel — Aggregate by days of the week (to NY) — Week of {week_dates[0]}", fontsize=14, pad=12)
        else:
            plt.title(f"Lincoln Tunnel — Aggregate by days of the week (to NY) — Weeks of {', '.join(week_dates)}", fontsize=14, pad=12)
    elif week_range:
        plt.title(f"Lincoln Tunnel — Aggregate by days of the week (to NY) ({week_range[0]} to {week_range[1]})", fontsize=14, pad=12)
    else:
        plt.title("Lincoln Tunnel — Aggregate by days of the week (to NY)", fontsize=14, pad=12)
    plt.xlabel("Time of Day (New York)")
    plt.ylabel("Minutes (To NY)")
    _set_hour_ticks()
    _set_y_ticks_from5_every3([agg["time_to_ny"]])
    _apply_grid()

    from matplotlib.lines import Line2D
    present_days = [d for d in weekday_order if not agg[agg["weekday"] == d].empty]
    handles = [Line2D([0], [0], color=weekday_colors[d], lw=1.2, label=d) for d in present_days]

    plt.legend(handles=handles, title="Weekday", ncol=2, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)

    # Determine output filename for NY direction
    if week_dates:
        if len(week_dates) == 1:
            tag = f"week_{week_dates[0]}"
        else:
            tag = "weeks_" + "_".join(week_dates)
        out_png_ny = os.path.join(OUT_DIR, f"aggregate_to_ny_{tag}.png")
    else:
        out_png_ny = os.path.join(OUT_DIR, "aggregate_to_ny.png")
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
        # Removed scatter: plt.scatter(xv, yv, s=4, alpha=1.0, label="_nolegend_", color=weekday_colors.get(day))
    # Title logic for To NJ
    if week_dates:
        if len(week_dates) == 1:
            plt.title(f"Lincoln Tunnel — Aggregate by days of the week (to NJ) — Week of {week_dates[0]}", fontsize=14, pad=12)
        else:
            plt.title(f"Lincoln Tunnel — Aggregate by days of the week (to NJ) — Weeks of {', '.join(week_dates)}", fontsize=14, pad=12)
    elif week_range:
        plt.title(f"Lincoln Tunnel — Aggregate by days of the week (to NJ) ({week_range[0]} to {week_range[1]})", fontsize=14, pad=12)
    else:
        plt.title("Lincoln Tunnel — Aggregate by days of the week (to NJ)", fontsize=14, pad=12)
    plt.xlabel("Time of Day (New York)")
    plt.ylabel("Minutes (To NJ)")
    _set_hour_ticks()
    _set_y_ticks_from5_every3([agg["time_to_nj"]])
    _apply_grid()

    from matplotlib.lines import Line2D
    present_days = [d for d in weekday_order if not agg[agg["weekday"] == d].empty]
    handles = [Line2D([0], [0], color=weekday_colors[d], lw=1.2, label=d) for d in present_days]

    plt.legend(handles=handles, title="Weekday", ncol=2, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)

    # Determine output filename for NJ direction
    if week_dates:
        if len(week_dates) == 1:
            tag = f"week_{week_dates[0]}"
        else:
            tag = "weeks_" + "_".join(week_dates)
        out_png_nj = os.path.join(OUT_DIR, f"aggregate_to_nj_{tag}.png")
    else:
        out_png_nj = os.path.join(OUT_DIR, "aggregate_to_nj.png")
    plt.savefig(out_png_nj, dpi=160)
    print(f"Saved {out_png_nj}")


# --------------------- Comparison Plotting (Weekday vs Weekend, Holiday vs Non-holiday) ---------------------
def plot_compare(df, kind="weekday_weekend", week_range=None):
    """
    Compare plots similar to aggregate but contrasting two groups:
    - kind == "weekday_weekend": compares Weekday (Mon-Fri) vs Weekend (Sat-Sun)
    - kind == "holiday": compares Holiday vs Non-holiday using US federal holidays
    - kind == "monday_blocks": each Monday as a line, time-of-day blocks (rush vs non-rush)
    - kind == "time_of_day_blocks": lines for To NY and To NJ, all days, by time-of-day blocks
    - kind == weekday: each day (e.g., Monday) as a line
    - kind == "months": each month as a line
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Ensure helper columns exist
    df = df.copy()
    df["weekday"] = df["timestamp"].dt.day_name()
    df["time_bin"] = df["timestamp"].dt.floor("5min").dt.strftime("%H:%M")
    df["date"] = df["timestamp"].dt.date

    # ---- Time-of-day block definitions (refined for "blocks" and "time_of_day_blocks") ----
    block_edges = [0, 120, 360, 480, 540, 600, 660, 720, 960, 1020, 1080, 1140, 1200, 1320, 1440]
    block_labels = [
        "12–2am", "2–6am", "6–8am", "8–9am", "9–10am", "10–11am", "11am–12pm",
        "12–4pm", "4–5pm", "5–6pm", "6–7pm", "7–8pm", "8–10pm", "10pm–12am"
    ]

    def assign_time_of_day_block(minutes: int):
        for i in range(len(block_edges) - 1):
            if block_edges[i] <= minutes < block_edges[i + 1]:
                return block_labels[i]
        # If at exactly 1440 (shouldn't happen), assign to last block
        return block_labels[-1]

    if kind == "blocks":
        # Get weekday argument from caller (should be passed via function argument or closure, see main)
        # We'll use a global variable _compare_weekday if set; otherwise, error.
        if not hasattr(plot_compare, "_weekday") or plot_compare._weekday is None:
            print("[error] --compare blocks requires --weekday argument (e.g. --weekday monday)")
            return
        target_weekday = plot_compare._weekday.capitalize()
        filtered = df[df["weekday"] == target_weekday].copy()
        if filtered.empty:
            print(f"[warn] No data found for weekday {target_weekday}.")
            return
        filtered["minute"] = filtered["timestamp"].dt.hour * 60 + filtered["timestamp"].dt.minute
        filtered["block"] = filtered["minute"].apply(assign_time_of_day_block)
        filtered["group"] = filtered["date"].astype(str)
        agg = filtered.groupby(["group", "block"], observed=False)[["time_to_ny", "time_to_nj"]].mean(numeric_only=True).reset_index()
        agg["block"] = pd.Categorical(agg["block"], categories=block_labels, ordered=True)
        agg = agg.sort_values(["group", "block"])
        group_order = sorted(agg["group"].unique())
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        x = list(range(len(block_labels)))
        # Plot To NY
        plt.figure(figsize=(10, 6))
        for i, grp in enumerate(group_order):
            gdf = agg[agg["group"] == grp]
            if gdf.empty:
                continue
            # y-values aligned with block_labels
            y_vals = gdf.set_index("block").reindex(block_labels)["time_to_ny"].to_numpy()
            plt.plot(
                x,
                y_vals,
                label=grp,
                marker="o",
                linewidth=1.5,
                alpha=0.9,
                color=colors[i % len(colors)]
            )
        plt.title(f"Lincoln Tunnel — {target_weekday}s: Rush vs Non-Rush (To NY)", fontsize=14, pad=12)
        plt.xlabel("Time of Day Block")
        plt.ylabel("Average Minutes (To NY)")
        plt.xticks(x, block_labels, rotation=15)
        _set_y_ticks_from5_every3([agg["time_to_ny"]])
        _apply_grid()
        plt.legend(title=target_weekday, ncol=2, loc="upper left", frameon=True, framealpha=0.6)
        plt.tight_layout(pad=1.3)
        out_png = os.path.join(OUT_DIR, f"{target_weekday.lower()}_blocks_to_ny.png")
        plt.savefig(out_png, dpi=160)
        print(f"Saved {out_png}")
        # Plot To NJ
        plt.figure(figsize=(10, 6))
        for i, grp in enumerate(group_order):
            gdf = agg[agg["group"] == grp]
            if gdf.empty:
                continue
            y_vals = gdf.set_index("block").reindex(block_labels)["time_to_nj"].to_numpy()
            plt.plot(
                x,
                y_vals,
                label=grp,
                marker="o",
                linewidth=1.5,
                alpha=0.9,
                color=colors[i % len(colors)]
            )
        plt.title(f"Lincoln Tunnel — {target_weekday}s: Rush vs Non-Rush (To NJ)", fontsize=14, pad=12)
        plt.xlabel("Time of Day Block")
        plt.ylabel("Average Minutes (To NJ)")
        plt.xticks(x, block_labels, rotation=15)
        _set_y_ticks_from5_every3([agg["time_to_nj"]])
        _apply_grid()
        plt.legend(title=target_weekday, ncol=2, loc="upper left", frameon=True, framealpha=0.6)
        plt.tight_layout(pad=1.3)
        out_png = os.path.join(OUT_DIR, f"{target_weekday.lower()}_blocks_to_nj.png")
        plt.savefig(out_png, dpi=160)
        print(f"Saved {out_png}")
        return
    elif kind == "time_of_day_blocks":
        # All days, aggregate by block, separate lines for To NY and To NJ
        df2 = df.copy()
        df2["minute"] = df2["timestamp"].dt.hour * 60 + df2["timestamp"].dt.minute
        df2["block"] = df2["minute"].apply(assign_time_of_day_block)
        # Aggregate across all days, by block
        agg = df2.groupby("block", observed=False)[["time_to_ny", "time_to_nj"]].mean(numeric_only=True).reset_index()
        agg["block"] = pd.Categorical(agg["block"], categories=block_labels, ordered=True)
        agg = agg.sort_values("block")
        # Plot both directions as lines
        plt.figure(figsize=(10, 6))
        plt.plot(
            agg["block"],
            agg["time_to_ny"],
            marker="o",
            linewidth=1.8,
            color="C0",
            label="To NY"
        )
        plt.plot(
            agg["block"],
            agg["time_to_nj"],
            marker="o",
            linewidth=1.8,
            color="C1",
            label="To NJ"
        )
        plt.title("Lincoln Tunnel — Daily Time-of-Day Averages", fontsize=14, pad=12)
        plt.xlabel("Time of Day Block")
        plt.ylabel("Average Minutes")
        plt.xticks(block_labels, rotation=15)
        _set_y_ticks_from5_every3([agg["time_to_ny"], agg["time_to_nj"]])
        _apply_grid()
        plt.legend(ncol=2, loc="upper left", frameon=True, framealpha=0.6)
        plt.tight_layout(pad=1.3)
        out_png = os.path.join(OUT_DIR, "time_of_day_blocks.png")
        plt.savefig(out_png, dpi=160)
        print(f"Saved {out_png}")
        return
    # ---- Existing logic for other compare kinds ----
    if kind == "weekday_weekend":
        df["group"] = df["timestamp"].dt.dayofweek.apply(lambda d: "Weekend" if d >= 5 else "Weekday")
        title_suffix = "Weekday vs Weekend"
        fname_suffix = "weekday_vs_weekend"
    elif kind == "holiday":
        start = df["date"].min()
        end = df["date"].max()
        try:
            cal = USFederalHolidayCalendar()
            holidays = cal.holidays(start=start, end=end).to_pydatetime()
            holiday_dates = set([h.date() for h in holidays])
        except Exception:
            holiday_dates = set()
        df["group"] = df["date"].apply(lambda d: "Holiday" if d in holiday_dates else "Non-holiday")
        title_suffix = "Holiday vs Non-holiday"
        fname_suffix = "holiday_vs_nonholiday"
    elif kind.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        target_day = kind.capitalize()
        df = df[df["weekday"] == target_day]
        df["group"] = df["date"].astype(str)  # each day is its own group
        title_suffix = f"All {target_day}s"
        fname_suffix = f"compare_{target_day.lower()}"
    elif kind == "months":
        # group by month as "YYYY-MM"
        df["month_str"] = df["timestamp"].dt.to_period("M").astype(str)
        # Optionally: filter out months with insufficient data (e.g., < 5 days)
        month_counts = df.groupby("month_str")["date"].nunique()
        sufficient_months = month_counts[month_counts >= 5].index
        df = df[df["month_str"].isin(sufficient_months)]
        df["group"] = df["month_str"]
        title_suffix = "Monthly Comparison"
        fname_suffix = "compare_months"
    else:
        raise ValueError("Unsupported compare kind: choose 'weekday_weekend', 'holiday', 'months', 'monday_blocks', 'time_of_day_blocks', or a weekday name")

    df["time_to_ny"] = df["time_to_ny"].where(df["time_to_ny"] > 0)
    df["time_to_nj"] = df["time_to_nj"].where(df["time_to_nj"] > 0)

    agg = df.groupby(["group", "time_bin"], observed=False)[["time_to_ny", "time_to_nj"]].mean(numeric_only=True).reset_index()

    group_order = sorted(agg["group"].unique(), key=lambda x: (0 if "Week" in x or "Non" in x else 1, x))
    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    def _plot_direction(direction_col, ylabel, out_fname):
        plt.figure(figsize=(12, 7))
        for i, grp in enumerate(group_order):
            gdf = agg[agg["group"] == grp].copy()
            if gdf.empty:
                continue
            times = pd.to_datetime(gdf["time_bin"], format="%H:%M", errors="coerce")
            gdf["x_num"] = (times.dt.hour * 60 + times.dt.minute).astype(int)
            gdf = gdf.dropna(subset=["x_num"]).sort_values("x_num").drop_duplicates(subset=["x_num"]).reset_index(drop=True)
            if gdf.empty:
                continue
            plt.plot(
                gdf["x_num"],
                gdf[direction_col].astype(float),
                label=grp,
                linewidth=1.5,
                alpha=0.9,
                color=colors[i % len(colors)]
            )
            # Ensure no scatter points are drawn in compare plot
            # (no plt.scatter call)

        title = f"Lincoln Tunnel — {title_suffix} (Average {ylabel})"
        if week_range:
            title = f"{title} ({week_range[0]} to {week_range[1]})"
        plt.title(title, fontsize=14, pad=12)
        plt.xlabel("Time of Day (New York)")
        plt.ylabel(f"Minutes ({ylabel})")
        _set_hour_ticks()
        _set_y_ticks_from5_every3([agg[direction_col]])
        _apply_grid()
        plt.legend(title="", ncol=2, loc="upper left", frameon=True, framealpha=0.6)
        plt.tight_layout(pad=1.3)
        out_png = os.path.join(OUT_DIR, out_fname)
        plt.savefig(out_png, dpi=160)
        print(f"Saved {out_png}")

    _plot_direction("time_to_ny", "To NY", f"{fname_suffix}_to_ny.png")
    _plot_direction("time_to_nj", "To NJ", f"{fname_suffix}_to_nj.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD (defaults to most recent date in data)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), inclusive")
    parser.add_argument("--aggregate", action="store_true", help="also produce weekday median plots (Mon–Sun)")
    parser.add_argument("--week", action="append", help="Any date within the week (YYYY-MM-DD); can be repeated to include multiple weeks")
    parser.add_argument("--compare", choices=[
        "weekday_weekend", "holiday", "months",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "blocks", "time_of_day_blocks"
    ], help="Produce comparison plots.")
    parser.add_argument("--weekday", type=str, help="Used with --compare blocks: specify weekday (e.g. monday, tuesday, ...)")
    parser.add_argument("--blocks", action="store_true", help="In aggregate mode: plot weekday aggregate using refined time-of-day blocks")
    args = parser.parse_args()

    df = load_df()
    # Apply date filters if provided (affects all downstream plots)
    if args.start or args.end:
        df = apply_date_filter(df, start=args.start, end=args.end)

    # Aggregate logic (move above plot logic for day/range)
    if args.aggregate:
        compare_df = df
        compare_week_range = None
        week_dates = None
        if args.week:
            week_ranges = []
            dfs_to_merge = []
            for week_str in args.week:
                try:
                    week_date = datetime.strptime(week_str, "%Y-%m-%d").date()
                    week_start = week_date - timedelta(days=week_date.weekday())
                    week_end = week_start + timedelta(days=6)
                    df_week = apply_date_filter(df, start=week_start.isoformat(), end=week_end.isoformat())
                    if not df_week.empty:
                        dfs_to_merge.append(df_week)
                        week_ranges.append((week_start, week_end))
                    else:
                        print(f"[warn] No data for week of {week_start} to {week_end}, skipping.")
                except ValueError:
                    raise SystemExit(f"Invalid --week: {week_str}. Use YYYY-MM-DD.")
            if dfs_to_merge:
                combined_df = pd.concat(dfs_to_merge, ignore_index=True)
                full_start = min(start for start, _ in week_ranges)
                full_end = max(end for _, end in week_ranges)
                week_dates = [str(w[0]) for w in week_ranges]
                plot_aggregate(combined_df, week_range=(full_start, full_end), week_dates=week_dates)
                compare_week_range = (full_start, full_end)
                # For blocks: use combined_df and week_dates
                if args.blocks:
                    plot_aggregate_blocks(combined_df, week_range=(full_start, full_end), week_dates=week_dates)
            else:
                print("[warn] No non-empty data found for specified weeks.")
        else:
            plot_aggregate(df)
            if args.blocks:
                plot_aggregate_blocks(df, week_range=None, week_dates=None)

        if args.compare:
            # Attach weekday argument for blocks mode
            if args.compare == "blocks":
                if not args.weekday or args.weekday.lower() not in [
                    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
                ]:
                    raise SystemExit("--compare blocks requires --weekday (monday, tuesday, ..., sunday)")
                plot_compare._weekday = args.weekday.lower()
            else:
                plot_compare._weekday = None
            plot_compare(df, kind=args.compare, week_range=compare_week_range)
            return
        else:
            return

    # Handle --compare even if --aggregate is not passed
    if args.compare:
        if args.compare == "blocks":
            if not args.weekday or args.weekday.lower() not in [
                "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
            ]:
                raise SystemExit("--compare blocks requires --weekday (monday, tuesday, ..., sunday)")
            plot_compare._weekday = args.weekday.lower()
        else:
            plot_compare._weekday = None
        plot_compare(df, kind=args.compare)
        return

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


# --- New function: plot_aggregate_blocks ---
def plot_aggregate_blocks(df, week_range=None, week_dates=None):
    """
    Plots aggregate by weekday and refined time-of-day blocks (same as those used in compare blocks).
    For each weekday, calculates average congestion for each block.
    Generates one plot each for To NY and To NJ.
    """
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    # Add weekday names
    df = df.copy()
    df["weekday"] = df["timestamp"].dt.day_name()
    # Define block edges and labels (same as compare blocks)
    block_edges = [0, 120, 360, 480, 540, 600, 660, 720, 960, 1020, 1080, 1140, 1200, 1320, 1440]
    block_labels = [
        "12–2am", "2–6am", "6–8am", "8–9am", "9–10am", "10–11am", "11am–12pm",
        "12–4pm", "4–5pm", "5–6pm", "6–7pm", "7–8pm", "8–10pm", "10pm–12am"
    ]

    def assign_time_of_day_block(minutes: int):
        for i in range(len(block_edges) - 1):
            if block_edges[i] <= minutes < block_edges[i + 1]:
                return block_labels[i]
        return block_labels[-1]

    # Assign block for each row
    df["minute"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["block"] = df["minute"].apply(assign_time_of_day_block)
    # Only keep valid values
    df["time_to_ny"] = df["time_to_ny"].where(df["time_to_ny"] > 0)
    df["time_to_nj"] = df["time_to_nj"].where(df["time_to_nj"] > 0)
    # Group by weekday and block
    agg = df.groupby(["weekday", "block"], observed=False)[["time_to_ny", "time_to_nj"]].mean(numeric_only=True).reset_index()
    # Ensure categorical order for plotting (set block as categorical BEFORE any filtering)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg["weekday"] = pd.Categorical(agg["weekday"], categories=weekday_order, ordered=True)
    # Set block as categorical and ordered IMMEDIATELY after agg is created (before any filtering)
    agg["block"] = pd.Categorical(agg["block"], categories=block_labels, ordered=True)
    agg = agg.sort_values(["weekday", "block"])
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
    x = list(range(len(block_labels)))
    for day in weekday_order:
        day_df = agg[agg["weekday"] == day].copy()
        if day_df.empty:
            continue
        # Ensure block is categorical and sorted for this day's slice
        day_df["block"] = pd.Categorical(day_df["block"], categories=block_labels, ordered=True)
        day_df = day_df.sort_values("block")
        # y-values aligned with block_labels
        y_vals = day_df.set_index("block").reindex(block_labels)["time_to_ny"].to_numpy()
        plt.plot(
            x,
            y_vals,
            label=day,
            marker="o",
            linewidth=1.5,
            color=weekday_colors.get(day)
        )
    # Title logic for To NY
    if week_dates:
        if len(week_dates) == 1:
            plt.title(f"Lincoln Tunnel — Aggregate by weekday & block (to NY) — Week of {week_dates[0]}", fontsize=14, pad=12)
        else:
            plt.title(f"Lincoln Tunnel — Aggregate by weekday & block (to NY) — Weeks of {', '.join(week_dates)}", fontsize=14, pad=12)
    elif week_range:
        plt.title(f"Lincoln Tunnel — Aggregate by weekday & block (to NY) ({week_range[0]} to {week_range[1]})", fontsize=14, pad=12)
    else:
        plt.title("Lincoln Tunnel — Aggregate by weekday & block (to NY)", fontsize=14, pad=12)
    plt.xlabel("Time of Day Block")
    plt.ylabel("Average Minutes (To NY)")
    plt.xticks(x, block_labels, rotation=15)
    _set_y_ticks_from5_every3([agg["time_to_ny"]])
    _apply_grid()
    from matplotlib.lines import Line2D
    present_days = [d for d in weekday_order if not agg[agg["weekday"] == d].empty]
    handles = [Line2D([0], [0], color=weekday_colors[d], lw=1.5, label=d) for d in present_days]
    plt.legend(handles=handles, title="Weekday", ncol=2, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)
    out_png_ny = os.path.join(OUT_DIR, "aggregate_blocks_to_ny.png")
    plt.savefig(out_png_ny, dpi=160)
    print(f"Saved {out_png_ny}")
    # Plot To NJ
    plt.figure(figsize=(12, 7))
    x = list(range(len(block_labels)))
    for day in weekday_order:
        day_df = agg[agg["weekday"] == day].copy()
        if day_df.empty:
            continue
        # Ensure block is categorical and sorted for this day's slice
        day_df["block"] = pd.Categorical(day_df["block"], categories=block_labels, ordered=True)
        day_df = day_df.sort_values("block")
        y_vals = day_df.set_index("block").reindex(block_labels)["time_to_nj"].to_numpy()
        plt.plot(
            x,
            y_vals,
            label=day,
            marker="o",
            linewidth=1.5,
            color=weekday_colors.get(day)
        )
    # Title logic for To NJ
    if week_dates:
        if len(week_dates) == 1:
            plt.title(f"Lincoln Tunnel — Aggregate by weekday & block (to NJ) — Week of {week_dates[0]}", fontsize=14, pad=12)
        else:
            plt.title(f"Lincoln Tunnel — Aggregate by weekday & block (to NJ) — Weeks of {', '.join(week_dates)}", fontsize=14, pad=12)
    elif week_range:
        plt.title(f"Lincoln Tunnel — Aggregate by weekday & block (to NJ) ({week_range[0]} to {week_range[1]})", fontsize=14, pad=12)
    else:
        plt.title("Lincoln Tunnel — Aggregate by weekday & block (to NJ)", fontsize=14, pad=12)
    plt.xlabel("Time of Day Block")
    plt.ylabel("Average Minutes (To NJ)")
    plt.xticks(x, block_labels, rotation=15)
    _set_y_ticks_from5_every3([agg["time_to_nj"]])
    _apply_grid()
    handles = [Line2D([0], [0], color=weekday_colors[d], lw=1.5, label=d) for d in present_days]
    plt.legend(handles=handles, title="Weekday", ncol=2, loc="upper left", frameon=True, framealpha=0.6, borderaxespad=0.4)
    plt.tight_layout(pad=1.3)
    out_png_nj = os.path.join(OUT_DIR, "aggregate_blocks_to_nj.png")
    plt.savefig(out_png_nj, dpi=160)
    print(f"Saved {out_png_nj}")


if __name__ == "__main__":
    main()
