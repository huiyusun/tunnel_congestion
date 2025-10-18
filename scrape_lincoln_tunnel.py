#!/usr/bin/env python3
"""
Lincoln Tunnel crossing-time scraper (direction-labeled, NY time).

- Loads the dedicated Lincoln Tunnel page only.
- Extracts minutes & mph for each direction and labels as to_ny / to_nj.
- Timestamps are in America/New_York at minute precision (YYYY-MM-DDTHH:MM).
- Writes to a stable, absolute CSV path in ./data/.

Output CSV: data/lincoln_tunnel_crossing_times.csv
Columns:
  timestamp, time_to_ny, time_to_nj, speed_to_ny, speed_to_nj, notes
"""

import csv
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from time import sleep

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

import tempfile
import shutil
from urllib.parse import urlparse

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "data")
OUT_CSV = os.path.join(OUT_DIR, "lincoln_tunnel_crossing_times.csv")

# ---------- Targets ----------
PAGE_URL = "https://www.panynj.gov/bridges-tunnels/en/lincoln-tunnel.html"

# ---------- Regex helpers ----------
# Direction label cues within panels
NY_LABELS = re.compile(r"\b(to\s*ny|ny[- ]?bound|manhattan[- ]?bound|to\s*manhattan|to\s*new\s*york)\b", re.I)
NJ_LABELS = re.compile(r"\b(to\s*nj|nj[- ]?bound|to\s*new\s*jersey|to\s*jersey|weehawken|hoboken|secaucus)\b", re.I)

# Minutes: "12 min", "12min", "12 mins"
MIN_RE = re.compile(r"(\d+)\s*min?s?\b", re.I)
# Speed: "35 mph", "35mi/h"
SPEED_RE = re.compile(r"(\d+)\s*(?:mph|mi/?h)\b", re.I)

# Aliases for robust DOM matching
DIRECTION_ALIASES_NY = [
    "to ny", "to new york", "nybound", "ny-bound", "manhattan-bound", "to manhattan"
]
DIRECTION_ALIASES_NJ = [
    "to nj", "to new jersey", "njbound", "nj-bound", "to jersey"
]


# ---------- CSV ----------
def ensure_csv():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "time_to_ny", "time_to_nj", "speed_to_ny", "speed_to_nj", "notes"])


def _read_last_timestamp(csv_path: str):
    """Return the first column (timestamp) of the last non-empty line, or None."""
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            if pos == 0:
                return None
            buf = bytearray()
            while pos > 0:
                pos -= 1
                f.seek(pos)
                b = f.read(1)
                if b == b"\n" and buf:
                    break
                if b != b"\n":
                    buf.extend(b)
            line = bytes(reversed(buf)).decode("utf-8", errors="ignore").strip()
            if not line:
                return None
            parts = line.split(",")
            return parts[0] if parts else None
    except Exception:
        return None


def _append_row_atomic(csv_path: str, row: list):
    """Atomically append a row to CSV by writing a temp file then replacing the original."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    dir_ = os.path.dirname(csv_path)
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, newline="") as tmp:
        tmp_path = tmp.name
        # Copy existing file first (if any)
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="") as src:
                shutil.copyfileobj(src, tmp)
        w = csv.writer(tmp)
        w.writerow(row)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, csv_path)


def _validate_minutes(m):
    """Return int minutes if within [5, 180], else None."""
    if m is None:
        return None
    try:
        m = int(m)
    except Exception:
        return None
    return m if 5 <= m <= 180 else None


# ---------- Text parsers (fallbacks) ----------
def extract_direction_minutes(text: str):
    """
    Heuristic minutes → direction from plain text. Returns (time_to_ny, time_to_nj, notes).
    """
    s = " ".join(text.split())
    time_to_ny, time_to_nj = None, None
    notes = []

    def nearest_min_after(idx):
        m = MIN_RE.search(s, idx)
        return int(m.group(1)) if m else None

    ny_iter = list(NY_LABELS.finditer(s))
    nj_iter = list(NJ_LABELS.finditer(s))

    if ny_iter:
        time_to_ny = nearest_min_after(ny_iter[0].start())
    if nj_iter:
        time_to_nj = nearest_min_after(nj_iter[0].start())

    if time_to_ny is None or time_to_nj is None:
        mins = [int(m.group(1)) for m in MIN_RE.finditer(s)]
        if len(mins) >= 2:
            if time_to_ny is None:
                time_to_ny = mins[0];
                notes.append("heuristic_to_ny_first_value")
            if time_to_nj is None:
                time_to_nj = mins[1];
                notes.append("heuristic_to_nj_second_value")
        elif len(mins) == 1 and time_to_ny is None and time_to_nj is None:
            time_to_ny = mins[0];
            notes.append("only_one_min_value_visible")

    return time_to_ny, time_to_nj, (";".join(notes) if notes else "ok")


def extract_speeds(text: str):
    """
    Heuristic speeds → direction from plain text. Returns (speed_to_ny, speed_to_nj).
    """
    s = " ".join(text.split())
    s_ny, s_nj = None, None

    def nearest_speed_after(idx):
        m = SPEED_RE.search(s, idx)
        return int(m.group(1)) if m else None

    ny_iter = list(NY_LABELS.finditer(s))
    nj_iter = list(NJ_LABELS.finditer(s))

    if ny_iter:
        s_ny = nearest_speed_after(ny_iter[0].start())
    if nj_iter:
        s_nj = nearest_speed_after(nj_iter[0].start())

    if s_ny is None or s_nj is None:
        speeds = [int(m.group(1)) for m in SPEED_RE.finditer(s)]
        if len(speeds) >= 2:
            if s_ny is None: s_ny = speeds[0]
            if s_nj is None: s_nj = speeds[1]
        elif len(speeds) == 1 and s_ny is None and s_nj is None:
            s_ny = speeds[0]

    return s_ny, s_nj


# ---------- DOM helpers ----------
def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _has_any_label(text: str, labels: list[str]) -> bool:
    t = _normalize(text)
    return any(lbl in t for lbl in labels)


def _find_panel_for_labels(page, labels: list[str], exclude_labels: list[str]):
    """
    Find the smallest DOM panel that contains any of `labels`,
    excludes any of `exclude_labels`, and includes a minutes/speed value.
    """

    def acceptable(text: str) -> bool:
        return (_has_any_label(text, labels)
                and not _has_any_label(text, exclude_labels)
                and (MIN_RE.search(text) or SPEED_RE.search(text)))

    # Strategy 1: common containers with :has-text
    candidates = page.locator("section:has-text(\"To \"), article:has-text(\"To \"), div:has-text(\"To \")")
    count = min(candidates.count(), 100)
    for i in range(count):
        loc = candidates.nth(i)
        try:
            t = loc.inner_text(timeout=800)
            if acceptable(t):
                return loc
        except Exception:
            continue

    # Strategy 2: start at the label text & walk up
    for lbl in labels:
        base = page.locator(f"text=/{re.escape(lbl)}/i").first
        if base and base.count() > 0:
            for depth in range(1, 12):
                try:
                    cont = base.locator(f"xpath=ancestor::*[{depth}]").nth(0)
                    t = cont.inner_text(timeout=800)
                    if acceptable(t):
                        return cont
                except Exception:
                    continue

    # Strategy 3: role-based generic regions
    roles = page.locator("div[role='region'], div[role='group'], section, article")
    count = min(roles.count(), 150)
    for i in range(count):
        loc = roles.nth(i)
        try:
            t = loc.inner_text(timeout=600)
            if acceptable(t):
                return loc
        except Exception:
            continue

    return None


def _parse_panel_text(txt: str, label_regex: re.Pattern):
    """
    Extract minutes and mph from a single direction panel's text,
    preferring values that appear *after* the direction label.
    """
    start_idx = None
    for m in label_regex.finditer(txt):
        start_idx = m.start()
        break

    if start_idx is None:
        mins = MIN_RE.search(txt)
        mph = SPEED_RE.search(txt)
    else:
        mins = MIN_RE.search(txt, start_idx)
        mph = SPEED_RE.search(txt, start_idx)

    m = int(mins.group(1)) if mins else None
    s = int(mph.group(1)) if mph else None
    return m, s


def extract_from_dom(page):
    """
    DOM-first extraction with retries. Returns (time_to_ny, time_to_nj, speed_to_ny, speed_to_nj, notes).
    """
    notes = []
    t_ny = t_nj = s_ny = s_nj = None

    # Retry up to ~5s for dynamic tiles
    for _ in range(5):
        ny_panel = _find_panel_for_labels(page, DIRECTION_ALIASES_NY, DIRECTION_ALIASES_NJ)
        nj_panel = _find_panel_for_labels(page, DIRECTION_ALIASES_NJ, DIRECTION_ALIASES_NY)

        # Parse NY
        if ny_panel:
            try:
                tny = ny_panel.inner_text(timeout=1200)
                m, s = _parse_panel_text(tny, NY_LABELS)
                if m is not None: t_ny = m
                if s is not None: s_ny = s
            except Exception:
                pass

        # Parse NJ
        if nj_panel:
            try:
                tnj = nj_panel.inner_text(timeout=1200)
                m, s = _parse_panel_text(tnj, NJ_LABELS)
                if m is not None: t_nj = m
                if s is not None: s_nj = s
            except Exception:
                pass

        if t_ny is not None and t_nj is not None and s_ny is not None and s_nj is not None:
            notes.append("dom_ok")
            break

        sleep(1.0)

    # Fallback to text parsing if needed
    if t_ny is None or t_nj is None or s_ny is None or s_nj is None:
        try:
            body = page.inner_text("body", timeout=4000)
        except Exception:
            body = ""
        m_ny, m_nj, note = extract_direction_minutes(body)
        s_ny2, s_nj2 = extract_speeds(body)

        if t_ny is None: t_ny = m_ny
        if t_nj is None: t_nj = m_nj
        if s_ny is None: s_ny = s_ny2
        if s_nj is None: s_nj = s_nj2

        notes.append("fallback_text_parse;" + (note or ""))

    return t_ny, t_nj, s_ny, s_nj, (";".join(notes) if notes else "ok")


# ---------- Scrape ----------
def scrape_once(timeout_ms=60000):
    tz = ZoneInfo("America/New_York")
    timestamp_ny = datetime.now(tz).strftime("%Y-%m-%dT%H:%M")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        # Block common third-party trackers/fonts to speed up load; allow panynj.gov
        blocked_substrings = [
            "googletagmanager", "google-analytics", "doubleclick", "facebook", "twitter",
            "fonts.googleapis.com", "fonts.gstatic.com", "hotjar", "segment.io", "newrelic",
        ]

        def _should_block(url: str) -> bool:
            try:
                host = (urlparse(url).hostname or "").lower()
            except Exception:
                host = ""
            if host.endswith("panynj.gov"):
                return False
            u = url.lower()
            return any(s in u for s in blocked_substrings)

        ctx.route("**/*", lambda route: route.abort() if _should_block(route.request.url) else route.continue_())

        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)

        # Load the page and give it breathing room
        page.goto(PAGE_URL, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)

        # Extra readiness wait: ensure at least one "XX min" appears; nudge lazy-loaded content
        try:
            page.wait_for_selector("text=/\\b\\d+\\s*min\\b/i", timeout=8000)
        except Exception:
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            except Exception:
                pass
            page.wait_for_timeout(1200)
        # Best-effort: dismiss common cookie/consent overlays if present
        for sel in [
            "text=/accept all/i",
            "text=/accept/i",
            "text=/agree/i",
            "role=button[name=/accept/i]",
            "role=button[name=/ok/i]",
        ]:
            try:
                btn = page.locator(sel).first
                if btn and btn.count() > 0:
                    btn.click(timeout=800)
                    break
            except Exception:
                continue

        time_to_ny, time_to_nj, speed_to_ny, speed_to_nj, notes = extract_from_dom(page)

        browser.close()

    return {
        "timestamp": timestamp_ny,
        "time_to_ny": time_to_ny,
        "time_to_nj": time_to_nj,
        "speed_to_ny": speed_to_ny,
        "speed_to_nj": speed_to_nj,
        "notes": notes,
    }


# ---------- Main ----------
def main():
    ensure_csv()
    try:
        rec = scrape_once()
    except PWTimeoutError as e:
        print(f"TimeoutError while scraping (1st attempt): {e}. Retrying once with 120s timeout...")
        sleep(10)
        try:
            rec = scrape_once(timeout_ms=120000)
        except PWTimeoutError as e2:
            print(f"TimeoutError on retry (120s): {e2}. Skipping this run.")
            return
    except Exception as e:
        print(f"Unhandled error while scraping: {e.__class__.__name__}: {e}")
        return

    # Schema validation for minutes (5–180). Keep speeds as-is (may be None).
    original_notes = rec.get("notes", "ok") or "ok"
    add_notes = []
    ny_valid = _validate_minutes(rec["time_to_ny"])
    nj_valid = _validate_minutes(rec["time_to_nj"])
    if ny_valid is None and rec["time_to_ny"] is not None:
        add_notes.append("invalid_time_to_ny")
    if nj_valid is None and rec["time_to_nj"] is not None:
        add_notes.append("invalid_time_to_nj")
    rec["time_to_ny"], rec["time_to_nj"] = ny_valid, nj_valid
    if add_notes:
        rec["notes"] = original_notes + ";" + ";".join(add_notes)

    # Success path: append one row to CSV, but skip if everything is missing
    all_missing = (
            rec["time_to_ny"] is None and rec["time_to_nj"] is None and
            rec["speed_to_ny"] is None and rec["speed_to_nj"] is None
    )
    if all_missing:
        print(
            f"No values found (minutes/speeds) at {rec['timestamp']} — skipping CSV write. Notes={rec['notes']}"
        )
        return

    # De-dupe by timestamp: skip if last line already has this timestamp
    last_ts = _read_last_timestamp(OUT_CSV)
    if last_ts == rec["timestamp"]:
        print(f"Duplicate timestamp {rec['timestamp']} — skipping CSV write.")
        return

    # Atomic append of the row
    _append_row_atomic(OUT_CSV, [
        rec["timestamp"],
        rec["time_to_ny"],
        rec["time_to_nj"],
        rec["speed_to_ny"],
        rec["speed_to_nj"],
        rec["notes"],
    ])

    print(
        f"Logged time_to_ny={rec['time_to_ny']} min, time_to_nj={rec['time_to_nj']} min, "
        f"speed_to_ny={rec['speed_to_ny']} mph, speed_to_nj={rec['speed_to_nj']} mph "
        f"at {rec['timestamp']} (NY time)"
    )


if __name__ == "__main__":
    main()
