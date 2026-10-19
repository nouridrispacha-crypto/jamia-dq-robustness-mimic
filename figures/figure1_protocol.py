# -*- coding: utf-8 -*-
"""
Figure 1: schematic four-step protocol overview.
Includes an automated post-render overlap check on every text object so
layout bugs are caught before the file is written, instead of relying on
hand-estimated coordinates.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT_IMG = Path(r"C:\Users\IDRINO\Desktop\JAMIA_PAPER\img")
OUT_IMG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans"})

BOX_COLORS = ["#152634", "#E6F0F9", "#F7EFDA", "#E6F0E1"]
TEXT_COLORS = ["#FFFFFF", "#152634", "#152634", "#152634"]
BADGE_COLORS = ["#1F3A52", "#1F3A52", "#B8791F", "#3F6B3A"]
BAND_FILL = "#F0E7F4"
BAND_EDGE = "#9B6FAE"
BAND_INK = "#3E2A4A"
BAND_MUTED = "#6B4F79"
ARROW_COLOR = "#8C97A0"
MUTED = "#5B6670"

fig, ax = plt.subplots(figsize=(11.6, 7.6))
ax.set_xlim(0, 11.6)
ax.set_ylim(0, 7.6)
ax.axis("off")
fig.patch.set_facecolor("white")

box_w, box_h = 2.35, 3.15
xs = [0.35, 2.95, 5.55, 8.15]
y0 = 3.75

# box 2 is titled generically because scenario S2-MD, listed in box 3,
# degrades all five dimensions rather than one
titles = ["Clean data", "Degrade the data", "Choose when", "Measure the drop"]

text_objects = []  # (label, Text) for overlap checking

for i, x in enumerate(xs):
    box = FancyBboxPatch((x, y0), box_w, box_h,
                           boxstyle="round,pad=0.02,rounding_size=0.16",
                           linewidth=0, facecolor=BOX_COLORS[i], zorder=2)
    ax.add_patch(box)

    bx, by = x + 0.05, y0 + box_h - 0.05
    badge = Circle((bx, by), 0.24, facecolor=BADGE_COLORS[i], edgecolor="white",
                    linewidth=2, zorder=4)
    ax.add_patch(badge)
    ax.text(bx, by, str(i + 1), ha="center", va="center", fontsize=14,
            fontweight="bold", color="white", zorder=5)

    t = ax.text(x + box_w / 2, y0 + box_h - 0.65, titles[i], ha="center", va="top",
                fontsize=14, fontweight="bold", color=TEXT_COLORS[i], zorder=4,
                linespacing=1.3)
    text_objects.append((f"title{i}", t))

for x0, x1 in zip(xs[:-1], xs[1:]):
    arrow = FancyArrowPatch((x0 + box_w + 0.06, y0 + box_h / 2), (x1 - 0.06, y0 + box_h / 2),
                              arrowstyle="-|>", mutation_scale=20, linewidth=2.0,
                              color=ARROW_COLOR, zorder=3)
    ax.add_patch(arrow)

# ---- step 1 body ---------------------------------------------------------
x = xs[0]
t = ax.text(x + box_w / 2, y0 + 1.55, "MIMIC-IV ICU", ha="center", va="center",
            fontsize=11, color="white", zorder=4)
text_objects.append(("s1a", t))
t = ax.text(x + box_w / 2, y0 + 1.10, "18 or 19 features, first 24 h", ha="center", va="center",
            fontsize=10, color="#C9D4DD", zorder=4)
text_objects.append(("s1b", t))
# TODO(N-sweep): restore 'N = 1500 / 3500 / 7000' once continuous and time-to-event runs are complete
t = ax.text(x + box_w / 2, y0 + 0.55, "N = 3500\n(binary: 1500, 3500, 7000)", ha="center", va="center",
            fontsize=10, linespacing=1.5, color="#C9D4DD", zorder=4)
text_objects.append(("s1c", t))

# ---- step 2 body ---------------------------------------------------------
x = xs[1]
t = ax.text(x + box_w / 2, y0 + 1.75, "completeness, coherence,\nvalidity, precision,\nuniqueness",
            ha="center", va="center", fontsize=9.5, color="#33475A", zorder=4,
            linespacing=1.5)
text_objects.append(("s2a", t))
t = ax.text(x + box_w / 2, y0 + 1.05, "one dimension at a time,\nor all five under S2-MD",
            ha="center", va="center", fontsize=8.5, style="italic",
            color="#4A6B8A", zorder=4, linespacing=1.4)
text_objects.append(("s2c", t))
t = ax.text(x + box_w / 2, y0 + 0.45, "corruption level\n$c = 0 \\rightarrow 0.3$",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="#152634", zorder=4, linespacing=1.5)
text_objects.append(("s2b", t))

# ---- step 3 body ---------------------------------------------------------
x = xs[2]
t = ax.text(x + box_w / 2, y0 + 2.02, "S1  train only", ha="center", va="center",
            fontsize=11.5, color="#33291A", zorder=4)
text_objects.append(("s3a", t))
t = ax.text(x + box_w / 2, y0 + 1.64, "S2  test only", ha="center", va="center",
            fontsize=11.5, color="#33291A", zorder=4)
text_objects.append(("s3b", t))
t = ax.text(x + box_w / 2, y0 + 1.34, "deployment realism", ha="center", va="center",
            fontsize=8.5, style="italic", color="#8A6A2E", zorder=4)
text_objects.append(("s3c", t))
t = ax.text(x + box_w / 2, y0 + 0.96, "S3  both (joint)", ha="center", va="center",
            fontsize=11.5, color="#33291A", zorder=4)
text_objects.append(("s3d", t))
t = ax.text(x + box_w / 2, y0 + 0.55, "S2-MD  all five at once", ha="center",
            va="center", fontsize=11.5, color="#33291A", zorder=4)
text_objects.append(("s3e", t))
t = ax.text(x + box_w / 2, y0 + 0.24, "test time, binary outcome", ha="center",
            va="center", fontsize=8.5, style="italic", color="#8A6A2E", zorder=4)
text_objects.append(("s3f", t))

# ---- step 4 body: mini declining-performance chart -----------------------
x = xs[3]
cx0, cw = x + 0.40, box_w - 0.80
cy0, ch = y0 + 0.45, 1.15
xcurve = np.linspace(0, 1, 40)
ycurve = 1 - 0.65 * xcurve ** 2.1
ax.plot(cx0 + xcurve * cw, cy0 + ycurve * ch, color="#3F6B3A", lw=2.4, zorder=4)
ax.annotate("", xy=(cx0, cy0 + ch + 0.35), xytext=(cx0, cy0 - 0.05),
            arrowprops=dict(arrowstyle="-|>", color="#5B7A55", lw=1.3), zorder=4)
ax.annotate("", xy=(cx0 + cw + 0.12, cy0), xytext=(cx0, cy0),
            arrowprops=dict(arrowstyle="-|>", color="#5B7A55", lw=1.3), zorder=4)
t = ax.text(cx0 - 0.08, cy0 + ch + 0.37, "perf.", ha="left", va="bottom", fontsize=9,
            color="#3F6B3A", zorder=4)
text_objects.append(("s4a", t))
t = ax.text(cx0 + cw + 0.14, cy0 - 0.10, "$c$", ha="left", va="top", fontsize=10,
            style="italic", color="#3F6B3A", zorder=4)
text_objects.append(("s4b", t))

# ---- bracket under all four steps: the whole protocol is run per task -----
band_y0, band_h = 1.35, 1.45
BAND_W = xs[3] + box_w - 0.35                     # band aligned on the four steps
lx, rx = xs[0], xs[3] + box_w                       # span of the four steps
yb, tick = y0 - 0.22, 0.14
ax.plot([lx, lx, rx, rx], [yb + tick, yb, yb, yb + tick], color=ARROW_COLOR, lw=1.8,
        solid_capstyle="round", zorder=3)
mid = (lx + rx) / 2
ax.add_patch(FancyArrowPatch((mid, yb), (mid, band_y0 + band_h + 0.06), arrowstyle="-|>",
                             mutation_scale=20, linewidth=1.8, color=ARROW_COLOR, zorder=3))

# ---- bottom band: repeated for 3 tasks ------------------------------------
band = FancyBboxPatch((0.35, band_y0), BAND_W, band_h,
                        boxstyle="round,pad=0.02,rounding_size=0.14",
                        linewidth=1.6, edgecolor=BAND_EDGE, facecolor=BAND_FILL, zorder=2)
ax.add_patch(band)
t = ax.text(0.35 + BAND_W / 2, band_y0 + band_h - 0.28,
            "Run for three prediction tasks, each with its own models and metrics",
            ha="center", va="top", fontsize=13, fontweight="bold", color=BAND_INK, zorder=4)
text_objects.append(("banda", t))

tasks = [
    ("Binary: mortality", "AUROC, Brier, F1-macro"),
    ("Continuous: ICU-LOS", "RMSE, R$^2$, MAE"),
    ("Time-to-event: death", "C-index, IBS"),
]
task_xs = [0.35 + BAND_W * frac for frac in (0.19, 0.5, 0.81)]
for j, ((title, metric), tx) in enumerate(zip(tasks, task_xs)):
    t = ax.text(tx, band_y0 + band_h - 0.74, title, ha="center", va="top", fontsize=11.5,
                fontweight="bold", color=BAND_INK, zorder=4)
    text_objects.append((f"task{j}a", t))
    t = ax.text(tx, band_y0 + band_h - 1.06, metric, ha="center", va="top", fontsize=10,
                style="italic", color=BAND_MUTED, zorder=4)
    text_objects.append((f"task{j}b", t))

for frac in (0.345, 0.655):   # light dividers between the three tasks
    ax.plot([0.35 + BAND_W * frac] * 2, [band_y0 + 0.18, band_y0 + band_h - 0.62],
            color="#D9C7E3", lw=1.0, zorder=3)

# ---- footer ---------------------------------------------------------------
t = ax.text(0.35 + BAND_W / 2, band_y0 - 0.20,
            "Evaluation: bootstrap .632 (stratified for mortality), 500 replicates, 3 seeds (neural)",
            ha="center", va="top", fontsize=9.5, color=MUTED, zorder=4)
text_objects.append(("foot1", t))
t = ax.text(0.35 + BAND_W / 2, band_y0 - 0.48,
            "Wilcoxon signed-rank tests with Holm correction",
            ha="center", va="top", fontsize=9.5, color=MUTED, zorder=4)
text_objects.append(("foot2", t))

plt.tight_layout()

# ---- automated overlap check (within-box groups only; skip pairs known to
# sit in different boxes / bands, since those never collide by construction) --
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
boxes = {}
for label, t in text_objects:
    bb = t.get_window_extent(renderer=renderer)
    boxes[label] = bb

GROUPS = {
    "step1": ["title0", "s1a", "s1b", "s1c"],
    "step2": ["title1", "s2a", "s2b", "s2c"],
    "step3": ["title2", "s3a", "s3b", "s3c", "s3d", "s3e", "s3f"],
    "step4": ["title3", "s4a", "s4b"],
    "band": ["banda"] + [f"task{j}{ab}" for j in range(3) for ab in "ab"],
    "footer": ["foot1", "foot2"],
}

overlaps_found = []
for gname, labels in GROUPS.items():
    for a, b in combinations(labels, 2):
        if boxes[a].overlaps(boxes[b]):
            overlaps_found.append((gname, a, b))

if overlaps_found:
    print("OVERLAP CHECK FAILED:")
    for gname, a, b in overlaps_found:
        print(f"  [{gname}] {a} overlaps {b}")
else:
    print("Overlap check passed: no text collisions detected.")

for letter, xpos in zip("ABCD", [None]):
    pass

for ext in ("pdf", "png"):
    plt.savefig(OUT_IMG / f"figure1_protocol.{ext}", dpi=220, bbox_inches="tight",
                facecolor="white")
print(f"Figure saved : {OUT_IMG / 'figure1_protocol.pdf'}")
plt.close(fig)
