"""What did the watch see? Reads watch_summary.csv (one line per minute, written by web_app.py in the work folder) and prints how many
minutes were watched, how many alarms there were, and limits that the quiet minutes suggest.

    python3 night_report.py hw3_work/watch_summary.csv
    python3 night_report.py hw3_work/watch_summary.csv --hours 22:00-07:00        # only the minutes of the night

Run it on a night in which NOBODY was in the room: then every room alert and every change is false, and the largest quiet values are the floor
under which a limit must not be set. A limit is only a guess until a night like this has been recorded.
"""
import argparse
import csv
import sys


def minutes_of(label: str) -> int:
    h, m = label.split(" ")[1].split(":")
    return int(h) * 60 + int(m)


def in_window(label: str, a: int, b: int) -> bool:
    m = minutes_of(label)
    return a <= m < b if a < b else (m >= a or m < b)


def pct(v: list, q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def report(rows: list, hours=None) -> str:
    """The text of the report for the rows of the file (dicts); hours = (start, end) in minutes of the day or None."""
    if hours:
        rows = [r for r in rows if in_window(r["minute"], *hours)]
    if not rows:
        return "no minutes to report (is the window right, was the watch running?)"
    n = len(rows)
    alerts = sum(int(r["room_alerts"] or 0) for r in rows)
    anomalies = sum(int(r["anomalies"] or 0) for r in rows)
    changed = sum(1 for r in rows if int(r["changed"] or 0) > 0)
    errors = sum(int(r["errors"] or 0) for r in rows)
    quiet = [r for r in rows if not int(r["room_alerts"] or 0) and not int(r["anomalies"] or 0)]
    out = [f"minutes watched: {n} ({n / 60:.1f} h), from {rows[0]['minute']} to {rows[-1]['minute']}",
           f"room alerts: {alerts}   agent calls (instrument alarms): {anomalies}   minutes with a 'changed' sample: {changed}   samples with errors: {errors}",
           f"rate: {alerts * 1440.0 / n:.1f} room alerts and {anomalies * 1440.0 / n:.1f} instrument alarms per 24 h of watching"]
    ch = [x for r in quiet if (x := num(r["change_max"])) is not None]
    if ch:
        out.append(f"picture change (share of cells) in the quiet minutes: median {pct(ch, .5):.3f}, 99th percentile {pct(ch, .99):.3f}, largest {max(ch):.3f}"
                   f"   -> the limit SCENE_CHANGE must stay above {max(ch):.3f} (now 0.05); a safe choice is {max(0.05, 1.5 * max(ch)):.3f}")
    bj = [num(r["brightness_max"]) - num(r["brightness_min"]) for r in quiet if num(r["brightness_max"]) is not None and num(r["brightness_min"]) is not None]
    if bj:
        out.append(f"brightness spread inside a minute: 99th percentile {pct(bj, .99):.1f} grey levels, largest {max(bj):.1f} (the limit is 15 and 25 %)")
    nj = [num(r["noise_max"]) - num(r["noise_mean"]) for r in quiet if num(r["noise_max"]) is not None and num(r["noise_mean"]) is not None]
    if nj:
        out.append(f"noise peak above the mean of its minute: 99th percentile {pct(nj, .99):.1f} dB, largest {max(nj):.1f} dB (the limit is 8 dB over the reference)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_file")
    ap.add_argument("--hours", default=None, help="only the minutes inside this window of the day, for example 22:00-07:00")
    a = ap.parse_args()
    hours = None
    if a.hours:
        try:
            h1, m1, h2, m2 = (int(x) for x in a.hours.replace("-", ":").split(":"))
            hours = (h1 * 60 + m1, h2 * 60 + m2)
        except ValueError:
            print("--hours must look like 22:00-07:00", file=sys.stderr)
            return 2
    try:
        with open(a.csv_file, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as e:
        print(f"cannot read {a.csv_file}: {e}", file=sys.stderr)
        return 1
    print(report(rows, hours))
    return 0


if __name__ == "__main__":
    sys.exit(main())
