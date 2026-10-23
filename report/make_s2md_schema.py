# -*- coding: utf-8 -*-
"""Schematic of Scenario S2-MD (composite, test-time degradation), cumulative design.

Left: defect rates returned by the eHOP data-quality score.
Middle: the weights w_k derived from them (profile P_observed, validity raised to 0.9%).
Right: one test set corrupted by all five dimensions at once. The four cell-level
dimensions act first on distinct cells; uniqueness then overwrites whole records
with a copy of another record, defects included, so one patient
can carry several types of defect but no value carries two.

draw_s2md(ax) draws the panel in data coordinates 0..11.6 x 0..4.6; it is the
S2-MD construction figure of Section 3.4.2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import IMG
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans"})
IMG = Path(IMG)

DIMS = ["Completeness", "Precision", "Uniqueness", "Coherence", "Validity"]
RATES = [15.0, 2.8, 2.0, 1.9, 0.03]            # eHOP defect rates (%)
WEIGHTS = [68.5, 12.8, 9.1, 8.7, 0.9]          # P_observed (%), Supplementary Table S4
COLORS = ["#1B7F79", "#C9A227", "#7A4E2D", "#C2649A", "#5A5A5A"]   # distinct from the model colours
CLEAN = "#E9ECEF"
INK, MUTED, ARROW = "#152634", "#5B6670", "#8C97A0"
W, H = 11.6, 4.6


def _arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=ARROW, zorder=3))
    if label:
        ax.text((x0 + x1) / 2, y - 0.18, label, ha="center", va="top", fontsize=10.5,
                color=MUTED, style="italic", linespacing=1.3)


def _corrupted_grid(n_rows, n_cols, c, rng):
    """Cell labels: '' clean, else the dimension name (schematic of pollute_cumulative_duplast).

    The four cell-level dimensions act first on the records that are not copied;
    uniqueness then overwrites its target records with a copy of another record,
    defects included.
    """
    lab = np.full((n_rows, n_cols), "", dtype=object)
    budget = c * n_rows * n_cols
    w = dict(zip(DIMS, np.array(WEIGHTS) / 100))
    # uniqueness targets are drawn first but written last; they take no cell-level defect
    n_u = max(1, int(round(w["Uniqueness"] * budget / n_cols)))
    u_rows = rng.choice(n_rows, size=n_u, replace=False)
    free = np.ones((n_rows, n_cols), dtype=bool)             # cell-level reservations
    free[u_rows, :] = False
    # 2. coherence: pairs of adjacent cells
    n_pairs = int(round(w["Coherence"] * budget / 2))
    placed = 0
    while placed < n_pairs:
        r, k = rng.integers(0, n_rows), rng.integers(0, n_cols - 1)
        if r in u_rows:
            continue
        if free[r, k] and free[r, k + 1]:
            lab[r, k] = lab[r, k + 1] = "Coherence"
            free[r, k] = free[r, k + 1] = False
            placed += 1
    # 3. completeness: severity-weighted rows
    severe = rng.random(n_rows) < 0.17
    p_row = np.where(severe, 12.0, 1.0)
    n_comp = int(round(w["Completeness"] * budget))
    wts = (p_row[:, None] * free).ravel()
    pick = rng.choice(wts.size, size=n_comp, replace=False, p=wts / wts.sum())
    for f in pick:
        lab[f // n_cols, f % n_cols] = "Completeness"
        free[f // n_cols, f % n_cols] = False
    # 4-5. precision, then validity, on remaining free cells
    for d in ("Precision", "Validity"):
        k = max(1, int(round(w[d] * budget)))
        cand = np.argwhere(free)
        for r, col in cand[rng.choice(len(cand), size=k, replace=False)]:
            lab[r, col] = d
            free[r, col] = False
    # uniqueness last: each target record is overwritten by a copy of a source record
    n_defects = np.array([sum(lab[r, k] != "" for k in range(n_cols)) for r in range(n_rows)])
    pairs = []
    for t in u_rows:
        rich = [r for r in np.argsort(-n_defects)[:12] if r not in u_rows]
        s = int(min(rich, key=lambda r: abs(r - t)))   # a near source keeps the arrow short
        lab[t, :] = lab[s, :]
        pairs.append((int(t), s))
    return lab, u_rows, pairs


def draw_s2md(ax, c=0.3, seed=4):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ── 1. eHOP defect rates ──
    x0, top = 0.15, 3.85
    ax.text(x0, 4.35, "1  Defect rates in eHOP", fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(x0, 4.02, "quality score, 5,192 ICU stays", fontsize=10.5, color=MUTED, va="center")
    bar_x, bar_max = x0 + 1.75, 1.05
    for i, (d, r, col) in enumerate(zip(DIMS, RATES, COLORS)):
        y = top - 0.55 - i * 0.55
        ax.text(bar_x - 0.08, y, d, ha="right", va="center", fontsize=11, color=INK)
        ax.add_patch(Rectangle((bar_x, y - 0.15), max(r / 15.0 * bar_max, 0.02), 0.30,
                               facecolor=col, lw=0, zorder=2))
        ax.text(bar_x + max(r / 15.0 * bar_max, 0.02) + 0.06, y, f"{r:g}%", va="center",
                fontsize=10.5, color=MUTED)

    _arrow(ax, 3.6, 4.2, 2.05, "share of\nthe total")

    # ── 2. weights w_k ──
    x1 = 4.3
    ax.text(x1, 4.35, "2  Weights $w_k$", fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(x1, 4.02, "profile $P_{\\mathrm{observed}}$", fontsize=10.5, color=MUTED, va="center")
    col_x, col_w, col_bot, col_h = x1 + 0.2, 0.55, 0.45, 3.2
    y = col_bot + col_h
    for d, w, col in zip(DIMS, WEIGHTS, COLORS):
        h = max(w / 100 * col_h, 0.05)
        y -= h
        ax.add_patch(Rectangle((col_x, y), col_w, h, facecolor=col, edgecolor="white", lw=1, zorder=2))
        label_y = y + h / 2
        if d == "Validity":
            label_y = y - 0.16
        label = f"{d} {w:g}% (raised)" if d == "Validity" else f"{d} {w:g}%"
        ax.text(col_x + col_w + 0.1, label_y, label, va="center", fontsize=10.5, color=INK)

    _arrow(ax, 7.15, 7.75, 2.05, "apply to\nthe test\nset")

    # ── 3. one test set, all five dimensions at once ──
    x2 = 7.85
    ax.text(x2, 4.35, "3  Corrupt the test set", fontsize=14, fontweight="bold", color=INK, va="center")
    ax.text(x2, 4.02, "rows = patients, columns = variables (schematic)", fontsize=10.5,
            color=MUTED, va="center")
    n_rows, n_cols = 40, 14
    lab, u_rows, pairs = _corrupted_grid(n_rows, n_cols, c, np.random.default_rng(seed))
    colour = dict(zip(DIMS, COLORS))
    gx, gw, gbot, gh = x2 + 0.05, 1.95, 0.45, 3.2
    cw, rh = gw / n_cols, gh / n_rows
    for r in range(n_rows):
        for k in range(n_cols):
            ax.add_patch(Rectangle((gx + k * cw, gbot + gh - (r + 1) * rh), cw, rh,
                                   facecolor=colour.get(lab[r, k], CLEAN),
                                   edgecolor="white", lw=0.35, zorder=2))
    ax.add_patch(Rectangle((gx, gbot), gw, gh, fill=False, edgecolor=INK, lw=0.8, zorder=3))
    def _row_y(r):
        return gbot + gh - (r + 0.5) * rh

    for t, s in pairs:   # the copy reproduces its source record, defects included
        for r, style in ((t, dict(lw=2.4, ls="-")), (s, dict(lw=1.2, ls=(0, (2, 1.6))))):
            ax.add_patch(Rectangle((gx - 0.03, gbot + gh - (r + 1) * rh - 0.02), gw + 0.06, rh + 0.04,
                                   fill=False, edgecolor=colour["Uniqueness"], zorder=4, **style))
        ax.add_patch(FancyArrowPatch((gx + gw + 0.05, _row_y(s)), (gx + gw + 0.05, _row_y(t)),
                                     connectionstyle=f"arc3,rad={-0.6 if t > s else 0.6}",
                                     arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                     color=colour["Uniqueness"], zorder=5))

    lx = gx + gw + 0.18
    ax.add_patch(Rectangle((lx, 3.44), 0.22, 0.22, facecolor=CLEAN, lw=0))
    ax.text(lx + 0.3, 3.55, "clean value", va="center", fontsize=10.5, color=INK)
    ax.add_patch(Rectangle((lx, 3.09), 0.22, 0.22, facecolor=colour["Uniqueness"],
                           edgecolor=colour["Uniqueness"], lw=2.4))
    ax.text(lx + 0.3, 3.2, "copied record (row)", va="center", fontsize=10.5, color=INK)
    ax.add_patch(Rectangle((lx, 2.74), 0.22, 0.22, facecolor="white",
                           edgecolor=colour["Uniqueness"], lw=1.2, ls=(0, (2, 1.6))))
    ax.text(lx + 0.3, 2.85, "its source record", va="center", fontsize=10.5, color=INK)
    ax.text(lx, 2.42, "1. Four cell-level\ndimensions: one defect\nper cell, any patient",
            va="top", fontsize=10.5, color=INK, linespacing=1.35)
    ax.text(lx, 1.55, "2. Uniqueness last:\nrecords copied with\nthe defects they carry",
            va="top", fontsize=10.5, color=INK, linespacing=1.35)
    ax.text(lx, 0.7, f"~{int(c * 100)}% of values altered;\na patient can carry\nseveral defect types",
            va="top", fontsize=10.5, color=INK, fontweight="bold", linespacing=1.35)


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else IMG
    fig, ax = plt.subplots(figsize=(W, H))
    draw_s2md(ax)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figure_s2md_schema.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote", out / "figure_s2md_schema.png")
