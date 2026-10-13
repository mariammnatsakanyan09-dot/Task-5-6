"""Build the Task 2 section of the report (PDF) from a finished run.

    python make_report.py runs/<timestamp> --lecture-enhancement <number> [--lecture-macropixels <N>] \
        [--repo-url <url>] [--video <link or path>]

Reads experiments.json (what the tool returned for every experiment), state.json (the agent's final message),
run_info.json (model, request) and the exp_NN_images.png files of the run folder. Writes in the run folder:
  task2_section.pdf         the table of experiments, the camera images, the comparison with the lecture
  experiments_table.csv     the same table for a spreadsheet
Nothing is computed by a model: every number is copied from the tool's records (the lecture number is the one you pass).
Needs reportlab and Pillow (pip install reportlab pillow); the fonts come with matplotlib.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
                                XPreformatted)


def register_fonts():
    import matplotlib
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DV", str(d / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", str(d / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", str(d / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DV-M", str(d / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")


def styles():
    base = dict(fontName="DV", fontSize=9.2, leading=12.6, alignment=TA_LEFT, spaceAfter=4)
    return dict(
        h1=ParagraphStyle("h1", fontName="DV-B", fontSize=14, leading=18, spaceAfter=6),
        h2=ParagraphStyle("h2", fontName="DV-B", fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3),
        p=ParagraphStyle("p", **base),
        small=ParagraphStyle("small", **{**base, "fontSize": 8, "leading": 10.5, "textColor": colors.HexColor("#444444")}),
        cell=ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=9.4),
        cellb=ParagraphStyle("cellb", fontName="DV-B", fontSize=7.6, leading=9.4),
        quote=ParagraphStyle("quote", fontName="DV-M", fontSize=7.4, leading=9.6, leftIndent=8,
                             borderPadding=(3, 3, 3, 6), textColor=colors.HexColor("#222222")),
    )


def P(text, style):
    return Paragraph(escape(text), style)


def load(run: Path):
    exps = json.loads((run / "experiments.json").read_text()) if (run / "experiments.json").exists() else []
    state = json.loads((run / "state.json").read_text()) if (run / "state.json").exists() else {}
    info = json.loads((run / "run_info.json").read_text()) if (run / "run_info.json").exists() else {}
    return exps, state, info


def write_csv(run: Path, exps):
    cols = ["experiment", "macropixel_size", "n_macropixels", "focus_size", "bg_gap", "bg_width", "n_bg_regions", "iterations",
            "n_flip", "n_avg", "enhancement", "enhancement_initial", "enhancement_vs_far_field", "focus_gain",
            "background_ratio", "seconds", "seed"]
    with open(run / "experiments_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in exps:
            s = e["settings"]
            w.writerow([e["experiment"], s["macropixel_size"], e["n_macropixels"], s["focus_size"], s["bg_gap"], s["bg_width"],
                        s["n_bg_regions"], s["iterations"], s["n_flip"], s["n_avg"], e["enhancement"], e["enhancement_initial"],
                        e.get("enhancement_vs_far_field"), e["focus_gain"], e["background_ratio"], e["seconds"], s.get("seed")])


def experiments_table(exps, st, best_no):
    head = ["#", "macropixel (N)", "focus px", "ring gap / width / regions", "iterations", "flip / avg",
            "enhancement start -> final", "vs far field", "focus gain", "background ratio"]
    rows = [[P(h, st["cellb"]) for h in head]]
    for e in exps:
        s = e["settings"]
        mark = " *" if e["experiment"] == best_no else ""
        rows.append([P(str(e["experiment"]), st["cell"]), P(f"{s['macropixel_size']} ({e['n_macropixels']})", st["cell"]),
                     P(str(s["focus_size"]), st["cell"]), P(f"{s['bg_gap']} / {s['bg_width']} / {s['n_bg_regions']}", st["cell"]),
                     P(str(s["iterations"]), st["cell"]), P(f"{s['n_flip']} / {s['n_avg']}", st["cell"]),
                     P(f"{e['enhancement_initial']:.2f} -> {e['enhancement']:.1f}{mark}", st["cellb" if mark else "cell"]),
                     P("-" if e.get("enhancement_vs_far_field") is None else f"{e['enhancement_vs_far_field']:.1f}", st["cell"]),
                     P(f"{e['focus_gain']:.1f}", st["cell"]), P(f"{e['background_ratio']:.2f}", st["cell"])])
    widths = [0.6, 2.2, 1.4, 1.9, 1.95, 1.25, 2.6, 1.4, 1.4, 2.3]      # sum 17.0 cm; each is wider than its longest header word
    t = Table(rows, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")])]))
    return t


def image(path: Path, width_cm: float):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def comparison_facts(best, lec_eta, lec_n):
    """Every number of the comparison, computed here from the tool's record of the best experiment and the lecture
    numbers that were passed in. The PDF and comparison.md both use these values."""
    n = best["n_macropixels"]
    tb = n / (2 * math.pi)
    f = dict(n=n, tb=tb, eff=best["enhancement"] / tb, far=best.get("enhancement_vs_far_field"), lec_eta=lec_eta, lec_n=lec_n,
             ratio=None, rel=None, lec_eff=None, far_eff=None)
    if f["far"] is not None:
        f["far_eff"] = f["far"] / tb
    if lec_eta is not None:
        f["ratio"] = best["enhancement"] / lec_eta
        r = f["ratio"]
        f["rel"] = ("higher than" if r >= 1.1 else "about the same as" if r >= 0.9 else
                    "somewhat lower than" if r >= 0.7 else "lower than")
        if lec_n:
            f["lec_eff"] = lec_eta / (lec_n / (2 * math.pi))
    return f


def comparison_sentences(best, f, lec_note):
    txt = []
    if f["lec_eta"] is not None:
        txt.append(f"The best enhancement of the agent, {best['enhancement']:.1f} (experiment {best['experiment']}, {f['n']} macropixels), "
                   f"is {f['rel']} the lecture's {f['lec_eta']:g} (ratio {best['enhancement']:.1f} / {f['lec_eta']:g} = {f['ratio']:.2f}).")
        if f["lec_n"]:
            txt.append(f"The two numbers are for different numbers of macropixels, so the last column puts them on one scale: the "
                       f"enhancement divided by the usual estimate N/(2 pi) for N binary-amplitude macropixels "
                       f"(here {f['eff']:.2f} against {f['lec_eff']:.2f} in the lecture).")
    else:
        txt.append("The lecture's enhancement was not given to this script (--lecture-enhancement), so only this run's numbers are shown.")
    if lec_note:
        txt.append(lec_note)
    return txt


def what_it_is_made_of(best, f):
    far = f["far"]
    return (f"What the enhancement is made of in the best experiment: the mean focus intensity rose by a factor "
            f"{best['focus_gain']:.1f} (from {best['focus_mean_counts_initial']:.1f} to {best['focus_mean_counts']:.1f} counts) "
            f"and the mean background fell by a factor {1 / best['background_ratio']:.1f} (from "
            f"{best['background_mean_counts_initial']:.1f} to {best['background_mean_counts']:.1f} counts). The search keeps a "
            f"flip when the ratio focus/background rises, so it lowers the proximal background as well as raising the focus. "
            + (f"Against the camera area that the search never used (the far field) the same focus is {far:.1f} times brighter "
               f"than the background there, which is the enhancement of the focus without the lowered ring."
               if far is not None else ""))


def comparison(exps, best, lec_eta, lec_n, lec_note, st):
    out = []
    f = comparison_facts(best, lec_eta, lec_n)
    n, tb = f["n"], f["tb"]
    rows = [[P(h, st["cellb"]) for h in ("", "macropixels N", "enhancement", "textbook N/(2 pi)", "enhancement / (N/(2 pi))")]]
    if lec_eta is not None:
        if lec_n:
            rows.append([P("lecture", st["cell"]), P(str(lec_n), st["cell"]), P(f"{lec_eta:g}", st["cell"]),
                         P(f"{lec_n / (2 * math.pi):.1f}", st["cell"]), P(f"{f['lec_eff']:.2f}", st["cell"])])
        else:
            rows.append([P("lecture", st["cell"]), P("not given", st["cell"]), P(f"{lec_eta:g}", st["cell"]),
                         P("-", st["cell"]), P("-", st["cell"])])
    else:
        rows.append([P("lecture", st["cell"]), P("-", st["cell"]), P("not given", st["cell"]), P("-", st["cell"]), P("-", st["cell"])])
    rows.append([P(f"this run, best (exp. {best['experiment']}), ring background as in the task", st["cellb"]), P(str(n), st["cell"]),
                 P(f"{best['enhancement']:.1f}", st["cellb"]), P(f"{tb:.1f}", st["cell"]), P(f"{f['eff']:.2f}", st["cell"])])
    if f["far"] is not None:
        rows.append([P("same experiment, against the far field", st["cell"]), P(str(n), st["cell"]),
                     P(f"{f['far']:.1f}", st["cell"]), P(f"{tb:.1f}", st["cell"]), P(f"{f['far_eff']:.2f}", st["cell"])])
    t = Table(rows, colWidths=[5.4 * cm, 2.6 * cm, 2.6 * cm, 3.0 * cm, 3.4 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef5")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    out.append(t)
    out.append(Spacer(1, 4))
    out.append(P(" ".join(comparison_sentences(best, f, lec_note)), st["p"]))
    out.append(P(what_it_is_made_of(best, f), st["p"]))
    return out


def comparison_md(exps, best, lec_eta, lec_n, lec_note) -> str:
    """The same comparison as plain text, written next to the run so that it is visible in the repository."""
    f = comparison_facts(best, lec_eta, lec_n)
    lines = ["# Comparison with the focusing result from the lecture", "",
             "| | macropixels N | enhancement | textbook N/(2 pi) | enhancement / (N/(2 pi)) |", "|---|--:|--:|--:|--:|"]
    if lec_eta is not None:
        lines.append(f"| lecture | {lec_n or 'not given'} | {lec_eta:g} | "
                     f"{(lec_n / (2 * math.pi)) if lec_n else float('nan'):.1f} | "
                     f"{'-' if f['lec_eff'] is None else format(f['lec_eff'], '.2f')} |".replace("nan", "-"))
    lines.append(f"| this run, best (experiment {best['experiment']}), ring background as in the task | {f['n']} | "
                 f"{best['enhancement']:.1f} | {f['tb']:.1f} | {f['eff']:.2f} |")
    if f["far"] is not None:
        lines.append(f"| same experiment, against the far field | {f['n']} | {f['far']:.1f} | {f['tb']:.1f} | {f['far_eff']:.2f} |")
    lines += ["", " ".join(comparison_sentences(best, f, lec_note)), "", what_it_is_made_of(best, f), ""]
    return "\n".join(lines)


def build(run: Path, args):
    register_fonts()
    exps, state, info = load(run)
    if not exps:
        sys.exit(f"{run}: no experiments.json (or it is empty): the agent made no successful experiment.")
    st = styles()
    best = max(exps, key=lambda e: e["enhancement"])
    write_csv(run, exps)
    out_pdf = Path(args.out) if args.out else run / "task2_section.pdf"
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=2.0 * cm, rightMargin=2.0 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="Task 2. Focusing through a simulated bench")
    n_calls = state.get('n_calls', len(exps))
    f = []
    f.append(P("Task 2. Focusing through a simulated bench", st["h1"]))
    code = args.repo_url or "(repository link)"
    f.append(Paragraph(f"<b>Code:</b> {escape(code)}", st["p"]))
    if args.video:
        f.append(Paragraph(f"<b>Video:</b> {escape(args.video)}", st["p"]))
    model = info.get("model") or "(model id)"
    f.append(Paragraph(f"<b>Model:</b> {escape(str(model))} through the OpenRouter API (free key), the same model in all four roles.", st["p"]))
    f.append(Paragraph(
        "<b>What was built.</b> A simulated bench (laser, DMD with amplitude only, scattering medium, camera with shot and read noise; "
        "E_out = |T E_in|^2) and a tool <font name='DV-M'>focus_search</font> that runs the whole search of the task on it: macropixels, a "
        "random start, the intensity in the focus region and in the proximal background (a ring split into regions), the enhancement as "
        "focus against background, flipping macropixels and keeping the flips that raise the enhancement. The tool returns the "
        "enhancement, re-measured on the final pattern with fresh shots, and the camera images. It is added to the executor of the "
        "Homework 1, Task 6 system (orchestrator, scientist, general agent, executor on one LangGraph graph). I typed one request, "
        f"<i>\"{escape(info.get('request', 'Focus the beam.'))}\"</i>, and nothing else: the agent chose the macropixel size, the region sizes "
        f"and the number of iterations itself, called the tool {n_calls} {'time' if n_calls == 1 else 'times'} "
        f"({len(exps)} successful {'experiment' if len(exps) == 1 else 'experiments'}) in {state.get('decisions', '?')} orchestrator decisions, and kept the enhancement the "
        "tool returned.", st["p"]))

    f.append(P("Experiments the agent ran", st["h2"]))
    f.append(experiments_table(exps, st, best["experiment"]))
    f.append(P("Enhancement = mean focus intensity / mean of the background regions, on the final pattern, re-measured with 50 fresh "
               "shots; 'start' is the same number for the random starting pattern. 'vs far field' is the focus intensity divided by the "
               "mean intensity of the camera pixels outside the ring (never used by the search). 'focus gain' and 'background ratio' are "
               "the mean counts in the focus and in the background ring after the search divided by the same counts before it. "
               "* best experiment. "
               "Every number is copied from the tool's records (experiments.json).", st["small"]))

    if state.get("final_text"):
        f.append(P("What the agent concluded (final message of the orchestrator)", st["h2"]))
        f.append(Paragraph(escape(state["final_text"].strip()).replace("\n", "<br/>"), st["quote"]))

    f.append(P("Camera images", st["h2"]))
    f.append(P("Each figure: the camera frame for the random starting pattern, the frame after the search (log scale; cyan = focus "
               "region, dashed = background ring), the DMD pattern found, and the enhancement during the search.", st["small"]))
    for e in exps:
        png = run / e["figure"]
        if png.exists():
            f.append(KeepTogether([image(png, 17.0), Spacer(1, 3)]))

    f.append(P("Comparison with the focusing result from the lecture", st["h2"]))
    f += comparison(exps, best, args.lecture_enhancement, args.lecture_macropixels, args.lecture_note, st)

    if len(exps) > 1:
        by = {}
        for e in exps:
            by.setdefault(e["settings"]["macropixel_size"], []).append((e["enhancement"], e["settings"]["iterations"]))
        line = "; ".join(f"macropixel {b} ({(64 // b) ** 2} macropixels): " + ", ".join(f"{v:.1f} ({it} iterations)" for v, it in vs)
                         for b, vs in sorted(by.items(), reverse=True))
        f.append(P("Enhancement by macropixel size across the experiments: " + line + ".", st["p"]))

    doc.build(f)
    md = comparison_md(exps, best, args.lecture_enhancement, args.lecture_macropixels, args.lecture_note)
    (run / "comparison.md").write_text(md)
    print(f"wrote {out_pdf}\nwrote {run / 'experiments_table.csv'}\nwrote {run / 'comparison.md'}")
    print("\n" + "\n".join(comparison_sentences(best, comparison_facts(best, args.lecture_enhancement, args.lecture_macropixels),
                                                 args.lecture_note)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder, for example runs/20261014-101500")
    ap.add_argument("--lecture-enhancement", type=float, default=None, help="the enhancement reached in the lecture")
    ap.add_argument("--lecture-macropixels", type=int, default=None, help="number of macropixels (N) of the lecture's result")
    ap.add_argument("--lecture-note", default="", help="one more sentence about the lecture's setup, added to the comparison")
    ap.add_argument("--repo-url", default=None, help="link of the code in the repository")
    ap.add_argument("--video", default=None, help="link or path of the video in the repository")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.repo_url:
        print("WARNING: no --repo-url, so the PDF says 'Code: (repository link)'. Add --repo-url <link>.")
    if not args.video:
        print("WARNING: no --video, so the PDF has no 'Video:' line. Add --video <link or path>.")
    build(Path(args.run), args)


if __name__ == "__main__":
    main()
