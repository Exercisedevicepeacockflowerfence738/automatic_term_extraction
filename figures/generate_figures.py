"""
Figure generation for the Automatic Term Extraction (ATE) article.

Produces all figures and tables as PNG files in this directory.

Numbering follows the order the figures appear in the article, not the order
of the functions below (function names are historical).

Data sources:
  * Figures 2, 3, 4 and Table 3 -> computed live from the pipeline inlined below
  * Figures 5, 6, 7             -> published results from the ATE literature
  * Tables 1, 2, 4, 5           -> literature + author synthesis

Attribution is never drawn inside an image; each figure is credited in the
caption beneath its embed in ARTICLE_automatic_term_extraction.md.

Run:  python generate_figures.py
"""

import math
import os
import re
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# =============================================================================
# THE ATE PIPELINE  (inlined so this script is self-contained)
#
#   1. linguistic filter -> candidates()
#   2. unithood          -> c_value()     Frantzi, Ananiadou & Mima (2000)
#   3. termhood          -> weirdness()   Ahmad et al.
#   4. combine           -> extract_terms()
#
# The same code, cell by cell with commentary, is in
# automatic-term-extraction-from-scratch.ipynb
# =============================================================================

# ---------------------------------------------------------------------------
# 1. LINGUISTIC FILTER: candidate generation
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "as",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "into", "about",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "we", "they", "he", "she", "you", "i", "our",
    "their", "his", "her", "your", "my", "can", "could", "may", "might",
    "will", "would", "shall", "should", "must", "have", "has", "had", "do",
    "does", "did", "not", "no", "such", "which", "who", "whom", "whose",
    "when", "where", "while", "also", "more", "most", "some", "any", "each",
    "both", "other", "there", "here", "how", "what", "why", "very", "over",
    "under", "between", "through", "during", "after", "before", "because",
    "however", "thus", "therefore", "using", "used", "use", "based", "one",
    "two", "three", "many", "much", "well", "often", "still", "only", "even",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]+")
SENT_SPLIT_RE = re.compile(r"[.!?;:\n]+")

MIN_LEN, MAX_LEN = 1, 4  # candidate length in tokens


def tokenize(text):
    """Lowercase word tokens; everything else is a boundary."""
    return TOKEN_RE.findall(text.lower())


def candidates(text, min_len=MIN_LEN, max_len=MAX_LEN):
    """
    Stand-in for a POS-pattern filter: inside each stopword-free chunk of a
    sentence, emit every n-gram of length min_len..max_len.
    With spaCy available, replace this with (ADJ|NOUN)+NOUN pattern matching.
    """
    out = []
    for sentence in SENT_SPLIT_RE.split(text):
        chunk = []
        for token in tokenize(sentence) + [None]:  # None flushes the chunk
            if token is None or token in STOPWORDS or len(token) < 3:
                for n in range(min_len, max_len + 1):
                    for i in range(len(chunk) - n + 1):
                        out.append(tuple(chunk[i:i + n]))
                chunk = []
            else:
                chunk.append(token)
    return out


# ---------------------------------------------------------------------------
# 2. C-VALUE (Frantzi, Ananiadou & Mima, 2000)
# ---------------------------------------------------------------------------
#   C-value(a) = log2|a| * f(a)                       if a is not nested
#   C-value(a) = log2|a| * (f(a) - 1/|Ta| * sum f(b)) otherwise
# where |a| = length of a in tokens, f(a) = frequency of a,
# Ta = set of longer candidates that contain a.

def c_value(freqs, min_freq=2):
    """freqs: Counter of candidate tuple -> frequency. Returns {candidate: score}."""
    longer_containing = defaultdict(list)  # a -> [f(b) for b in Ta]
    by_len = defaultdict(list)
    for cand in freqs:
        by_len[len(cand)].append(cand)

    for length, shorter_list in by_len.items():
        for shorter in shorter_list:
            for longer_len in range(length + 1, MAX_LEN + 1):
                for longer in by_len.get(longer_len, ()):
                    if _contains(longer, shorter):
                        longer_containing[shorter].append(freqs[longer])

    scores = {}
    for cand, freq in freqs.items():
        if freq < min_freq:
            continue
        length_weight = math.log2(len(cand)) if len(cand) > 1 else 0.5
        nested_in = longer_containing.get(cand)
        if nested_in:
            score = length_weight * (freq - sum(nested_in) / len(nested_in))
        else:
            score = length_weight * freq
        if score > 0:
            scores[cand] = score
    return scores


def _contains(longer, shorter):
    """True if `shorter` appears as a contiguous subsequence of `longer`."""
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


# ---------------------------------------------------------------------------
# 3. WEIRDNESS (Ahmad et al.) — contrastive: technical vs. general corpus
# ---------------------------------------------------------------------------
#   weirdness(w) = (f_tech(w) / N_tech) / (f_gen(w) / N_gen)

def weirdness(tech_tokens, general_tokens, smoothing=0.5):
    f_tech, f_gen = Counter(tech_tokens), Counter(general_tokens)
    n_tech, n_gen = max(len(tech_tokens), 1), max(len(general_tokens), 1)
    scores = {}
    for word, freq in f_tech.items():
        tech_rate = freq / n_tech
        gen_rate = (f_gen[word] + smoothing) / n_gen
        scores[word] = tech_rate / gen_rate
    return scores


def candidate_weirdness(cand, word_weirdness):
    """Aggregate word-level weirdness over a multi-word candidate (geometric mean)."""
    vals = [max(word_weirdness.get(w, 1.0), 1e-6) for w in cand]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


# ---------------------------------------------------------------------------
# 4. PIPELINE
# ---------------------------------------------------------------------------

def extract_terms(technical_corpus, general_corpus, top_n=20, min_freq=2,
                  have_single_word=True):
    min_len = MIN_LEN if have_single_word else 2
    cands = candidates(technical_corpus, min_len=min_len)
    freqs = Counter(cands)

    cv = c_value(freqs, min_freq=min_freq)
    ww = weirdness(tokenize(technical_corpus), tokenize(general_corpus))

    results = []
    for cand, cv_score in cv.items():
        w_score = candidate_weirdness(cand, ww)
        # Combined termhood: unithood (C-value) x domain specificity (weirdness)
        termhood = cv_score * math.log1p(w_score)
        results.append({
            "term": " ".join(cand),
            "freq": freqs[cand],
            "c_value": round(cv_score, 3),
            "weirdness": round(w_score, 2),
            "termhood": round(termhood, 3),
        })

    results.sort(key=lambda r: r["termhood"], reverse=True)
    return results[:top_n]


# --- demonstration corpora ---------------------------------------------------
# Wind energy is one of the four ACTER benchmark domains.

TECHNICAL_CORPUS = """
Wind energy is converted into electrical energy by a wind turbine. The wind
turbine consists of a rotor, a nacelle and a tower. The rotor blades capture
kinetic energy from the wind and transfer it to the main shaft. A horizontal
axis wind turbine is the dominant design in modern wind farms, while the
vertical axis wind turbine remains rare. The gearbox increases the rotational
speed before the generator converts mechanical energy into electrical energy.
Pitch control adjusts the rotor blades to regulate the power output, and yaw
control aligns the nacelle with the wind direction. The capacity factor of a
wind farm describes the ratio of actual energy output to the rated power output
over a period. An offshore wind farm typically achieves a higher capacity
factor than an onshore wind farm because the wind speed offshore is higher and
more stable. Grid integration of wind power requires reactive power
compensation and accurate wind speed forecasting. The power curve of a wind
turbine maps wind speed to power output between the cut-in wind speed and the
cut-out wind speed. Wind turbine blades made of glass fibre reinforced polymer
dominate the market. Condition monitoring of the gearbox and the generator
reduces the maintenance cost of an offshore wind farm. A doubly fed induction
generator allows variable speed operation of the wind turbine. Levelised cost
of energy is the standard metric for comparing a wind farm with other energy
sources. The rated power of a modern offshore wind turbine exceeds fifteen
megawatts, and the rotor diameter exceeds two hundred metres.
"""

GENERAL_CORPUS = """
The company announced a new plan on Monday. People in the city said the change
would affect the daily life of many families. A spokesman said the government
would review the decision next year. The report described how the market
reacted to the news and what the main effects were. Several members of the
group asked for more time to study the document before the final vote. The
weather was good and the event continued through the afternoon. Energy prices
rose slightly, and the cost of living increased for most households. A local
school opened a new building this year. The team won the match after a long
season. Many readers wrote letters about the article. The design of the new
product was praised, and the power of social media played a role in the
campaign. The speed of the change surprised the older members of the community.
The control of the process stayed with the same department. The output of the
factory grew by ten percent. The average wind was light on the coast.
"""

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = HERE

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

C = {
    "linguistic":  "#5B8DB8",   # blue
    "statistical": "#E1A624",   # amber
    "hybrid":      "#5BAD8D",   # green
    "neural":      "#C0392B",   # red
    "llm":         "#8E44AD",   # purple
    "neutral":     "#7F8C8D",   # grey
    "dark":        "#2C3E50",
    "light":       "#ECF0F1",
}


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


def render_table(col_labels, rows, title, name, col_chars,
                 fontsize=9.5, row_colors=None):
    """
    Render a styled table to PNG with proper text wrapping.

    col_chars : maximum characters per line for each column; also sets the
                relative column widths. Text longer than this is wrapped,
                never clipped.
    """
    import textwrap
    from matplotlib.patches import Rectangle

    CHAR_W = fontsize * 0.0091      # inches per character (DejaVu Sans)
    LINE_H = fontsize * 0.0208      # inches per text line
    PAD_X, PAD_Y = 0.10, 0.085      # cell padding, inches

    # -- wrap every cell -----------------------------------------------------
    def wrap_row(cells):
        return [textwrap.wrap(str(t), width=col_chars[i]) or [""]
                for i, t in enumerate(cells)]

    header = wrap_row(col_labels)
    body = [wrap_row(r) for r in rows]

    col_w = [c * CHAR_W + 2 * PAD_X for c in col_chars]
    total_w = sum(col_w)

    def row_h(wrapped):
        return max(len(c) for c in wrapped) * LINE_H + 2 * PAD_Y

    header_h = row_h(header)
    body_h = [row_h(r) for r in body]
    total_h = header_h + sum(body_h)

    fig, ax = plt.subplots(figsize=(total_w + 0.3, total_h + 0.95))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    ax.grid(False)

    def draw_row(wrapped, y_top, height, facecolor, textcolor, weight):
        x = 0.0
        for i, lines in enumerate(wrapped):
            ax.add_patch(Rectangle((x, y_top - height), col_w[i], height,
                                   facecolor=facecolor, edgecolor="#D5DBDB",
                                   linewidth=0.8, zorder=1))
            ty = y_top - PAD_Y - LINE_H * 0.72
            for line in lines:
                ax.text(x + PAD_X, ty, line, fontsize=fontsize,
                        color=textcolor, weight=weight, va="center",
                        ha="left", zorder=2)
                ty -= LINE_H
            x += col_w[i]

    y = total_h
    draw_row(header, y, header_h, C["dark"], "white", "bold")
    y -= header_h

    for i, wrapped in enumerate(body):
        if row_colors is not None and i < len(row_colors) and row_colors[i]:
            fc = row_colors[i]
        else:
            fc = "#FBFCFC" if i % 2 == 0 else "#F2F4F4"
        draw_row(wrapped, y, body_h[i], fc, "#212F3C", "normal")
        y -= body_h[i]

    ax.set_title(title, fontsize=12.5, weight="bold", pad=16, loc="left",
                 color=C["dark"])
    save(fig, name)


# =============================================================================
# FORMULAS — rendered with matplotlib mathtext
# =============================================================================
FORMULA_BG = "#F7F9FA"


def _formula_canvas(width, height, accent):
    """
    Blank panel with a light background and a coloured left accent bar.

    Attribution is NOT drawn inside the image — every figure is credited in
    the caption beneath its embed in the article markdown.
    """
    from matplotlib.patches import Rectangle, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    ax.add_patch(FancyBboxPatch(
        (0.004, 0.03), 0.992, 0.94,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=FORMULA_BG, edgecolor="#E4E8EA", linewidth=1.2,
        zorder=0, transform=ax.transAxes))
    ax.add_patch(Rectangle((0.004, 0.03), 0.007, 0.94,
                           facecolor=accent, edgecolor="none", zorder=1,
                           transform=ax.transAxes))
    return fig, ax


def draw_brace(ax, x, y_lo, y_hi, width=0.014, color="#2C3E50", lw=1.7):
    """Draw a left curly brace spanning [y_lo, y_hi] at horizontal position x."""
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    ym = (y_lo + y_hi) / 2
    q = width
    verts = [
        (x + q, y_hi), (x, y_hi), (x + q, ym + (y_hi - ym) * 0.5), (x + q, ym),
        (x + q, ym), (x + q, ym - (ym - y_lo) * 0.5), (x, y_lo), (x + q, y_lo),
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO]
    # a smoother brace: two mirrored quadratic segments meeting at a tip
    verts = [
        (x + q, y_hi), (x + q * 0.15, y_hi), (x + q * 0.15, ym + q * 0.6),
        (x, ym),
        (x + q * 0.15, ym - q * 0.6), (x + q * 0.15, y_lo), (x + q, y_lo),
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.CURVE3, Path.CURVE3, Path.LINETO]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor="none",
                                    edgecolor=color, linewidth=lw,
                                    transform=ax.transAxes, zorder=3,
                                    capstyle="round"))


def formula_cvalue():
    fig, ax = _formula_canvas(11.4, 2.35, C["statistical"])

    ax.text(0.045, 0.50, r"$\mathrm{C\text{-}value}(a)\;=$", fontsize=19,
            va="center", ha="left", color="#212F3C", transform=ax.transAxes)

    draw_brace(ax, 0.245, 0.20, 0.80, width=0.022)

    ax.text(0.295, 0.705, r"$\log_2 |a| \cdot f(a)$", fontsize=19,
            va="center", ha="left", color="#212F3C", transform=ax.transAxes)
    ax.text(0.295, 0.285,
            r"$\log_2 |a| \cdot \left( f(a) \;-\; "
            r"\dfrac{1}{P(T_a)} \sum_{b\, \in\, T_a} f(b) \right)$",
            fontsize=19, va="center", ha="left", color="#212F3C",
            transform=ax.transAxes)

    ax.text(0.975, 0.705, "if $a$ is not nested", fontsize=12.5, va="center",
            ha="right", color="#5D6D7E", style="italic", transform=ax.transAxes)
    ax.text(0.975, 0.285, "otherwise", fontsize=12.5, va="center",
            ha="right", color="#5D6D7E", style="italic", transform=ax.transAxes)

    save(fig, "formula2_cvalue.png")


def formula_cvalue_terms():
    """Legend explaining every symbol in the C-value formula."""
    cols = ["Symbol", "Meaning"]
    rows = [
        ["a", "The candidate term under evaluation"],
        ["|a|", "Length of a in words"],
        ["f(a)", "Frequency of a in the corpus"],
        ["Tₐ", "Set of longer candidate terms that contain a"],
        ["P(Tₐ)", "Number of those longer candidates"],
    ]
    render_table(cols, rows,
                 "Symbols in the C-value formula",
                 "formula3_cvalue_symbols.png",
                 col_chars=[10, 46], fontsize=9.5)


def formula_weirdness():
    fig, ax = _formula_canvas(9.6, 1.9, C["linguistic"])
    ax.text(0.5, 0.52,
            r"$\mathrm{weirdness}(w) \;=\; "
            r"\dfrac{\;f_{\mathrm{tech}}(w) \,/\, N_{\mathrm{tech}}\;}"
            r"{\;f_{\mathrm{gen}}(w) \,/\, N_{\mathrm{gen}}\;}$",
            fontsize=21, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    save(fig, "formula5_weirdness.png")


def formula_ncvalue():
    fig, ax = _formula_canvas(11.0, 2.6, C["statistical"])

    ax.text(0.5, 0.80,
            r"$\mathrm{weight}(w) \;=\; \dfrac{t(w)}{n}$",
            fontsize=18, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    ax.text(0.5, 0.50,
            r"$\mathrm{C}_{\mathrm{context}}(a) \;=\; "
            r"\sum_{b\, \in\, C_a} f_a(b)\cdot \mathrm{weight}(b)$",
            fontsize=18, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    ax.text(0.5, 0.16,
            r"$\mathrm{NC\text{-}value}(a) \;=\; "
            r"0.8 \cdot \mathrm{C\text{-}value}(a) \;+\; "
            r"0.2 \cdot \mathrm{C}_{\mathrm{context}}(a)$",
            fontsize=18, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    save(fig, "formula4_ncvalue.png")


def formula_termhood():
    fig, ax = _formula_canvas(9.6, 1.75, C["hybrid"])
    ax.text(0.5, 0.52,
            r"$\mathrm{termhood}(a) \;=\; \mathrm{C\text{-}value}(a) "
            r"\;\cdot\; \log\!\left(1 + \mathrm{weirdness}(a)\right)$",
            fontsize=19, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    save(fig, "formula6_termhood.png")


def formula_metrics():
    fig, ax = _formula_canvas(12.6, 1.95, C["neutral"])
    ax.text(0.19, 0.52,
            r"$\mathrm{Precision} = \dfrac{TP}{TP + FP}$",
            fontsize=18, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    ax.text(0.50, 0.52,
            r"$\mathrm{Recall} = \dfrac{TP}{TP + FN}$",
            fontsize=18, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    ax.text(0.82, 0.52,
            r"$F_1 = \dfrac{2 \cdot P \cdot R}{P + R}$",
            fontsize=18, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    save(fig, "formula7_metrics.png")


def formula_tor():
    fig, ax = _formula_canvas(10.4, 2.0, C["llm"])
    ax.text(0.5, 0.56,
            r"$\mathrm{TOR} \;=\; \dfrac{1}{N_q}\sum_{q}\;"
            r"\dfrac{\left|\, G_q \cap R_q \,\right|}{\left|\, G_q \,\right|}$",
            fontsize=20, va="center", ha="center", color="#212F3C",
            transform=ax.transAxes)
    ax.text(0.5, 0.145,
            "$G_q$ = gold terms of query $q$    ·    "
            "$R_q$ = terms in the retrieved demonstrations",
            fontsize=11, va="center", ha="center", color="#5D6D7E",
            transform=ax.transAxes)
    save(fig, "formula8_term_overlap_ratio.png")


def formula_pos_patterns():
    fig, ax = _formula_canvas(12.0, 2.5, C["linguistic"])

    patterns = [
        (r"$\mathrm{Noun}^{+}\ \mathrm{Noun}$",
         "closed filter — highest precision"),
        (r"$(\mathrm{Adj}\,|\,\mathrm{Noun})^{+}\ \mathrm{Noun}$",
         "standard filter"),
        (r"$\left((\mathrm{Adj}|\mathrm{Noun})^{+} \;|\; "
         r"(\mathrm{Adj}|\mathrm{Noun})^{*}(\mathrm{Noun}\ \mathrm{Prep})?\,"
         r"(\mathrm{Adj}|\mathrm{Noun})^{*}\right)\ \mathrm{Noun}$",
         "open filter — highest recall"),
    ]
    ys = [0.78, 0.50, 0.20]
    for (expr, note), y in zip(patterns, ys):
        ax.text(0.035, y, expr, fontsize=15, va="center", ha="left",
                color="#212F3C", transform=ax.transAxes)
        ax.text(0.975, y, note, fontsize=11, va="center", ha="right",
                color="#5D6D7E", style="italic", transform=ax.transAxes)

    save(fig, "formula1_pos_filters.png")


# =============================================================================
# FIGURE 1 — The canonical hybrid ATE pipeline
# =============================================================================
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.grid(False)

    stages = [
        (0.15, "Domain\ncorpus", "Raw specialised text\n(unstructured)", C["neutral"]),
        (2.55, "1. Linguistic\nfilter", "POS patterns\n(ADJ|NOUN)+ NOUN\n→ term candidates", C["linguistic"]),
        (4.95, "2. Unithood\nranking", "C-value\nnested-term\ncorrection", C["statistical"]),
        (7.35, "3. Termhood\nranking", "Weirdness / TF-IDF\nvs. general corpus", C["statistical"]),
        (9.75, "4. Termbase", "Ranked, thresholded\ndomain term list", C["hybrid"]),
    ]

    box_w, box_h = 2.0, 2.1
    y = 1.5

    for x, title, sub, color in stages:
        box = FancyBboxPatch((x, y), box_w, box_h,
                             boxstyle="round,pad=0.06,rounding_size=0.12",
                             linewidth=2, edgecolor=color,
                             facecolor=color, alpha=0.13)
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h - 0.42, title, ha="center", va="center",
                fontsize=11.5, weight="bold", color=C["dark"])
        ax.text(x + box_w / 2, y + box_h / 2 - 0.42, sub, ha="center", va="center",
                fontsize=8.8, color="#34495E", linespacing=1.5)

    for x, *_ in stages[:-1]:
        arr = FancyArrowPatch((x + box_w + 0.05, y + box_h / 2),
                              (x + box_w + 0.35, y + box_h / 2),
                              arrowstyle="-|>", mutation_scale=17,
                              linewidth=2, color=C["dark"])
        ax.add_patch(arr)

    # annotation bands
    ax.annotate("", xy=(2.55, 1.15), xytext=(4.55, 1.15),
                arrowprops=dict(arrowstyle="-", lw=2.5, color=C["linguistic"]))
    ax.text(3.55, 0.80, "linguistic knowledge", ha="center", fontsize=9.5,
            color=C["linguistic"], style="italic", weight="bold")

    ax.annotate("", xy=(4.95, 1.15), xytext=(9.35, 1.15),
                arrowprops=dict(arrowstyle="-", lw=2.5, color=C["statistical"]))
    ax.text(7.15, 0.80, "statistical evidence", ha="center", fontsize=9.5,
            color=C["statistical"], style="italic", weight="bold")

    ax.text(6.0, 0.15,
            "Hybrid design — used by 50% of ATE studies reviewed in "
            "Blandón Andrade et al. (2026)",
            ha="center", fontsize=9, color="#566573")


    ax.set_title("Figure 1 — The canonical hybrid ATE pipeline",
                 fontsize=13.5, weight="bold", loc="left", color=C["dark"], pad=12)
    save(fig, "fig1_ate_pipeline.png")


# =============================================================================
# FIGURE 2 — Raw frequency vs. C-value: the nested-term correction (REAL DATA)
# =============================================================================
def fig2_frequency_vs_cvalue():
    rows = extract_terms(TECHNICAL_CORPUS, GENERAL_CORPUS, top_n=10,
                         have_single_word=False)
    rows = sorted(rows, key=lambda r: r["freq"], reverse=True)

    terms = [r["term"] for r in rows]
    freqs = [r["freq"] for r in rows]
    cvals = [r["c_value"] for r in rows]

    y = np.arange(len(terms))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(y + h / 2, freqs, height=h, color=C["neutral"], alpha=0.75,
            label="Raw frequency  f(a)", edgecolor="white")
    ax.barh(y - h / 2, cvals, height=h, color=C["statistical"],
            label="C-value(a)", edgecolor="white")

    for i, (f, cv) in enumerate(zip(freqs, cvals)):
        ax.text(f + 0.08, i + h / 2, f"{f}", va="center", fontsize=8.5,
                color="#566573")
        ax.text(cv + 0.08, i - h / 2, f"{cv:.2f}", va="center", fontsize=8.5,
                color="#7D6608", weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    ax.set_xlim(0, 7.6)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    ax.grid(axis="y", visible=False)

    # highlight the most-discounted candidate
    disc = [(i, r) for i, r in enumerate(rows)
            if r["freq"] > 0 and r["c_value"] / r["freq"] < 0.45]
    if disc:
        i, r = disc[0]
        ax.annotate("heavily discounted:\nalmost always occurs\ninside a longer term",
                    xy=(r["c_value"], i - h / 2), xytext=(2.6, i - 0.15),
                    fontsize=8.6, color=C["neural"], linespacing=1.4,
                    arrowprops=dict(arrowstyle="->", color=C["neural"], lw=1.3))

    ax.set_title("Figure 2 — Raw frequency vs. C-value on the wind-energy corpus\n"
                 "C-value subtracts the frequency a candidate only borrows from longer terms",
                 fontsize=12.5, weight="bold", loc="left", color=C["dark"], pad=14)
    save(fig, "fig2_frequency_vs_cvalue.png")


# =============================================================================
# FIGURE 3 — Unithood vs. termhood plane (REAL DATA)
# =============================================================================
def fig3_termhood_plane():
    rows = extract_terms(TECHNICAL_CORPUS, GENERAL_CORPUS, top_n=26,
                         have_single_word=True)

    fig, ax = plt.subplots(figsize=(9.6, 6.4))

    for r in rows:
        n_words = len(r["term"].split())
        color = {1: C["neutral"], 2: C["statistical"], 3: C["neural"]}.get(
            n_words, C["llm"])
        ax.scatter(r["c_value"], r["weirdness"],
                   s=42 + 26 * r["freq"], color=color, alpha=0.72,
                   edgecolor="white", linewidth=1.2, zorder=3)

    ax.set_xlim(-0.4, 13.2)
    ax.set_ylim(1.0, 12.6)

    # greedy label placement: try candidate offsets until no collision
    label_these = sorted(rows, key=lambda r: r["termhood"], reverse=True)[:12]
    placed = []          # (x0, y0, x1, y1) in data coords
    x_span, y_span = 13.6, 11.6
    offsets = [(0.22, 0.20), (0.22, -0.42), (-0.22, 0.20), (-0.22, -0.42),
               (0.22, 0.62), (0.22, -0.84), (-0.22, 0.62), (-0.22, -0.84)]

    for r in label_these:
        px, py = r["c_value"], r["weirdness"]
        w = 0.085 * x_span * (len(r["term"]) / 12) ** 0.72
        hgt = 0.040 * y_span
        for dx, dy in offsets:
            lx = px + dx if dx > 0 else px + dx - w
            ly = py + dy
            box = (lx, ly - hgt / 2, lx + w, ly + hgt / 2)
            if not any(box[0] < p[2] and box[2] > p[0] and
                       box[1] < p[3] and box[3] > p[1] for p in placed):
                break
        placed.append(box)
        ax.annotate(r["term"], (px, py), xytext=(lx if dx > 0 else lx + w, ly),
                    ha="left" if dx > 0 else "right", va="center",
                    fontsize=8.6, color="#34495E")

    ax.axvline(2.0, color=C["dark"], lw=1.1, ls="--", alpha=0.45)
    ax.axhline(5.0, color=C["dark"], lw=1.1, ls="--", alpha=0.45)

    ax.text(0.985, 0.045, "low unithood and low termhood → rejected",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.8,
            color="#566573", style="italic")
    ax.text(0.985, 0.965,
            "Valid multi-word terms cluster in the upper-right quadrant",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
            color="#566573", style="italic")

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C["neutral"], label="1 word",
                   markersize=9, markeredgecolor="white"),
        plt.Line2D([], [], marker="o", ls="", color=C["statistical"], label="2 words",
                   markersize=9, markeredgecolor="white"),
        plt.Line2D([], [], marker="o", ls="", color=C["neural"], label="3 words",
                   markersize=9, markeredgecolor="white"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9.5,
              title="candidate length", title_fontsize=9.5, loc="upper left")

    ax.set_xlabel("C-value  (unithood — is it a coherent unit?)")
    ax.set_ylabel("Weirdness  (termhood — is it domain-specific?)")
    ax.set_title("Figure 3 — The unithood/termhood plane\n"
                 "Marker area is proportional to corpus frequency",
                 fontsize=12.5, weight="bold", loc="left", color=C["dark"], pad=14)
    save(fig, "fig3_termhood_plane.png")


# =============================================================================
# FIGURE 4 — F1 on ACTER: non-neural -> neural  (Tran et al., 2023, Tables)
# =============================================================================
def fig4_acter_f1():
    # Reported F1 ranges on the ACTER heart-failure held-out test set
    # Source: Tran et al. (2023), arXiv:2301.06767, comparative results section.
    systems = [
        ("e-Terminology",              15.3, 21.4, "non-neural"),
        ("NYU",                        31.5, 31.5, "non-neural"),
        ("NMF",                        30.1, 33.5, "non-neural"),
        ("RACAI",                      39.3, 39.3, "non-neural"),
        ("HAMLET (152 features)",      54.2, 66.1, "non-neural"),
        ("XLM-R  sequence clf.",       45.2, 48.5, "neural"),
        ("mBART  seq2seq",             53.2, 65.2, "neural"),
        ("Monolingual BERT  token",    38.9, 66.8, "neural"),
        ("XLM-R  token clf.",          58.3, 69.6, "neural"),
        ("XLM-R  zero-shot x-ling.",   58.3, 69.8, "neural"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(systems))

    for i, (name, lo, hi, kind) in enumerate(systems):
        color = C["neural"] if kind == "neural" else C["statistical"]
        if hi > lo:
            ax.plot([lo, hi], [i, i], lw=7, color=color, alpha=0.32,
                    solid_capstyle="round", zorder=2)
        ax.scatter([lo], [i], s=58, color=color, zorder=3, edgecolor="white",
                   linewidth=1.2)
        ax.scatter([hi], [i], s=58, color=color, zorder=3, edgecolor="white",
                   linewidth=1.2)
        txt = f"{lo:.1f}" if hi == lo else f"{lo:.1f}–{hi:.1f}"
        ax.text(hi + 1.1, i, txt, va="center", fontsize=8.8, color="#566573")

    ax.axvline(66.1, color=C["dark"], ls="--", lw=1.2, alpha=0.55)
    ax.text(66.6, -0.92, "best non-neural result\n(HAMLET, 66.1)",
            ha="left", va="center", fontsize=8.6, color=C["dark"],
            style="italic", linespacing=1.4)

    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in systems], fontsize=9.8)
    ax.set_ylim(len(systems) - 0.4, -1.5)      # inverted, with headroom on top
    ax.set_xlabel("F1-score on the ACTER held-out heart-failure test set (%)")
    ax.set_xlim(10, 82)
    ax.grid(axis="y", visible=False)

    handles = [
        mpatches.Patch(color=C["statistical"], alpha=0.7,
                       label="Non-neural (feature engineering / unsupervised)"),
        mpatches.Patch(color=C["neural"], alpha=0.7,
                       label="Neural (Transformer-based)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower left")

    ax.set_title("Figure 6 — ATE performance on the ACTER benchmark\n"
                 "Ranges span three languages (EN/FR/NL) and two annotation "
                 "settings (terms only / terms + named entities)",
                 fontsize=12.5, weight="bold", loc="left", color=C["dark"], pad=14)
    save(fig, "fig6_acter_f1_by_system.png")


# =============================================================================
# FIGURE 5 — What the literature actually uses (Blandón Andrade et al., 2026)
# =============================================================================
def fig5_literature_statistics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.4),
                                   gridspec_kw={"wspace": 0.42})

    # -- panel A: approach distribution -------------------------------------
    labels = ["Hybrid (linguistic + statistical)", "Statistical only", "Linguistic only"]
    sizes = [50.0, 38.2, 11.8]
    colors = [C["hybrid"], C["statistical"], C["linguistic"]]

    wedges, _, autotexts = ax1.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        pctdistance=0.79,
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(11)
        t.set_weight("bold")
    ax1.legend(wedges, labels, frameon=False, fontsize=9.5,
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1)
    ax1.set_title("A. Methodological approach", fontsize=11.5, weight="bold",
                  color=C["dark"], pad=14)
    ax1.grid(False)

    # -- panel B: algorithm adoption ----------------------------------------
    algos = ["C-value\n(Frantzi et al., 2000)", "TF-IDF"]
    adopt = [50, 31]
    bars = ax2.bar(algos, adopt, color=[C["statistical"], C["neutral"]],
                   width=0.5, edgecolor="white", linewidth=2)
    for b, v in zip(bars, adopt):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1.4, f"{v}%",
                 ha="center", fontsize=13, weight="bold", color=C["dark"])
    ax2.set_ylim(0, 62)
    ax2.set_ylabel("Share of reviewed studies (%)")
    ax2.set_title("B. Most-adopted ranking algorithms", fontsize=11.5,
                  weight="bold", color=C["dark"], pad=14)
    ax2.grid(axis="x", visible=False)

    fig.suptitle("Figure 5 — What ATE research actually uses (113 papers, 2015–2022)",
                 fontsize=13.5, weight="bold", color=C["dark"], x=0.005, ha="left",
                 y=1.04)
    save(fig, "fig5_literature_statistics.png")


# =============================================================================
# FIGURE 6 — LLM in-context learning vs. fine-tuned PLM (Chun et al., 2025)
# =============================================================================
def fig6_llm_vs_plm():
    benchmarks = ["ACTER\n(cross-domain)", "ACLR2\n(in-domain)", "BCGM\n(in-domain)"]
    llm = [60.2, 80.7, 53.8]
    plm = [61.2, 87.4, 88.5]

    x = np.arange(len(benchmarks))
    w = 0.34

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    b1 = ax.bar(x - w / 2, llm, w, label="LLM, few-shot in-context learning\n(best of Llama-3.1 / Gemma-2 / Mistral-Nemo)",
                color=C["llm"], edgecolor="white", linewidth=1.8)
    b2 = ax.bar(x + w / 2, plm, w, label="Fine-tuned pretrained LM (task-specific)",
                color=C["neural"], edgecolor="white", linewidth=1.8)

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{b.get_height():.1f}", ha="center", fontsize=10,
                    weight="bold", color=C["dark"])

    # gap annotations
    for i, (a, p) in enumerate(zip(llm, plm)):
        gap = p - a
        ax.annotate(f"Δ {gap:+.1f}", xy=(i, max(a, p) + 8), ha="center",
                    fontsize=10, color=C["dark"] if gap < 10 else C["neural"],
                    weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10.5)
    ax.set_ylabel("F1-score (%)")
    ax.set_ylim(0, 122)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left",
              bbox_to_anchor=(-0.01, 1.005))
    ax.grid(axis="x", visible=False)

    ax.set_title("Figure 7 — Where LLMs win and where they do not\n"
                 "LLMs close the gap under domain shift; fine-tuned encoders "
                 "remain far ahead in-domain",
                 fontsize=12.5, weight="bold", loc="left", color=C["dark"], pad=14)
    save(fig, "fig7_llm_vs_plm.png")


# =============================================================================
# FIGURE 7 — Threshold and length effects (REAL DATA)
# =============================================================================
def fig7_threshold_and_length():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # -- panel A: candidates surviving the min_freq threshold ---------------
    freqs = Counter(candidates(TECHNICAL_CORPUS))
    thresholds = [1, 2, 3, 4, 5]
    counts = [len(c_value(freqs, min_freq=t)) for t in thresholds]

    ax1.plot(thresholds, counts, marker="o", markersize=9, lw=2.6,
             color=C["statistical"], markeredgecolor="white",
             markeredgewidth=1.8, zorder=3)
    for t, c in zip(thresholds, counts):
        ax1.annotate(f"{c}", (t, c), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9.5,
                     color=C["dark"], weight="bold")

    ax1.axvline(2, color=C["neural"], ls="--", lw=1.5, alpha=0.7)
    ax1.text(2.08, max(counts) * 0.72,
             "min_freq = 2\nused in the demo\n→ discards every\nsingleton term",
             fontsize=8.8, color=C["neural"], va="top")

    ax1.set_xticks(thresholds)
    ax1.set_xlabel("Minimum frequency threshold")
    ax1.set_ylabel("Surviving candidates")
    ax1.set_title("A. The recall cost of frequency thresholding",
                  fontsize=11.5, weight="bold", color=C["dark"], pad=12)

    # -- panel B: top-ranked candidates, coloured by length ------------------
    rows = extract_terms(TECHNICAL_CORPUS, GENERAL_CORPUS, top_n=12,
                         have_single_word=True)
    colors = [C["neutral"], C["statistical"], C["neural"], C["llm"]]

    names = [r["term"] for r in rows]
    scores = [r["termhood"] for r in rows]
    lens = [len(r["term"].split()) for r in rows]
    ypos = np.arange(len(rows))

    ax2.barh(ypos, scores, color=[colors[min(k - 1, 3)] for k in lens],
             height=0.66, edgecolor="white", linewidth=1.5)
    for yy, s in zip(ypos, scores):
        ax2.text(s + max(scores) * 0.015, yy, f"{s:.1f}", va="center",
                 fontsize=8.6, color="#566573")

    ax2.set_yticks(ypos)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Termhood score")
    ax2.set_xlim(0, max(scores) * 1.16)
    ax2.grid(axis="y", visible=False)

    n_uni = sum(1 for k in lens if k == 1)
    ax2.annotate(f"{n_uni} of the top {len(rows)} candidates\nare single words",
                 xy=(scores[0] * 0.62, 1.9), fontsize=9, color=C["neural"],
                 linespacing=1.4)

    handles = [
        mpatches.Patch(color=colors[0], label="1 word"),
        mpatches.Patch(color=colors[1], label="2 words"),
        mpatches.Patch(color=colors[2], label="3 words"),
    ]
    ax2.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    ax2.set_title("B. Single words crowd the top of the ranking",
                  fontsize=11.5, weight="bold", color=C["dark"], pad=12)

    fig.suptitle("Figure 4 — Two failure modes of the unsupervised pipeline",
                 fontsize=13.5, weight="bold", color=C["dark"], x=0.005,
                 ha="left", y=1.04)
    save(fig, "fig4_threshold_and_length.png")


# =============================================================================
# TABLE 1 — Keyword extraction vs. automatic term extraction
# =============================================================================
def table1_keywords_vs_terms():
    cols = ["Dimension", "Keyword extraction", "Automatic term extraction"]
    rows = [
        ["Question answered", "What is this document about?", "What vocabulary defines this domain?"],
        ["Unit of analysis", "A single document", "A domain corpus"],
        ["Candidate generation", "All n-grams, or stopword-delimited chunks", "POS-constrained patterns, e.g. (ADJ|NOUN)+ NOUN"],
        ["Ranking signal", "TF-IDF, TextRank, YAKE, KeyBERT", "C-value, weirdness, neural token classification"],
        ["Reference corpus", "Implicit (the same collection)", "Explicit general-language contrast corpus"],
        ["Nested terms", "Not modelled", "Central concern (C-value)"],
        ["Typical output", "5–10 keywords per document", "A ranked termbase for the domain"],
        ["Evaluation", "Largely qualitative", "Gold standard; precision, recall, F1 (ACTER)"],
    ]
    render_table(cols, rows,
                 "Table 1 — Keyword extraction and automatic term extraction "
                 "are different tasks",
                 "table1_keywords_vs_terms.png",
                 col_chars=[22, 42, 48], fontsize=9.5)


# =============================================================================
# TABLE 2 — The ACTER benchmark
# =============================================================================
def table2_acter():
    cols = ["Domain", "Code", "Files", "Sentences", "Tokens", "Role"]
    rows = [
        ["Corruption",    "corp", "12",     "1 977–2 002", "52 847–61 107", "Training (parallel corpus)"],
        ["Equitation",    "equi", "34–78",  "2 809–3 669", "60 119–63 870", "Training (comparable)"],
        ["Wind energy",   "wind", "2–8",    "3 356–6 638", "58 684–69 759", "Training / validation"],
        ["Heart failure", "htfl", "174–210", "2 177–2 880", "57 204–57 899", "Held-out test set"],
    ]
    render_table(cols, rows,
                 "Table 4 — Composition of the ACTER corpus\n"
                 "Three languages (English, French, Dutch); ranges span languages",
                 "table4_acter_composition.png",
                 col_chars=[14, 7, 9, 14, 16, 28], fontsize=9.5)


# =============================================================================
# TABLE 3 — Annotation categories
# =============================================================================
def table3_annotation_categories():
    cols = ["Category", "Definition (Rigouts Terryn et al., 2020)", "Example"]
    rows = [
        ["Specific Term", "Domain-specific and lexicon-specific; relevant to the domain and known only by domain experts", "doubly fed induction generator"],
        ["Common Term", "Domain-specific but generally known", "heart, patients"],
        ["Out-of-Domain Term", "Lexicon-specific but not domain-relevant", "confidence interval, p-value"],
        ["Named Entity", "Proper names of people, organisations, brands", "Siemens Gamesa"],
    ]
    render_table(cols, rows,
                 "Table 2 — The four annotation categories of ACTER\n"
                 "An operational answer to the question \"what counts as a term?\"",
                 "table2_annotation_categories.png",
                 col_chars=[20, 52, 30], fontsize=9.5)


# =============================================================================
# TABLE 4 — Extracted terms from the use case (REAL DATA)
# =============================================================================
def table4_extracted_terms():
    rows_data = extract_terms(TECHNICAL_CORPUS, GENERAL_CORPUS, top_n=10,
                              have_single_word=False)
    cols = ["Rank", "Extracted term", "f(a)", "C-value", "Weirdness", "Termhood",
            "Assessment"]

    verdict = {
        "wind turbine": "valid term",
        "wind farm": "valid term",
        "wind speed": "valid term",
        "offshore wind": "partial — boundary error",
        "offshore wind farm": "valid term",
        "axis wind turbine": "partial — should be 'horizontal axis wind turbine'",
        "electrical energy": "valid term",
        "power output": "valid term",
        "rotor blades": "valid term",
        "axis wind": "spurious",
    }
    row_colors = []
    rows = []
    for i, r in enumerate(rows_data, 1):
        v = verdict.get(r["term"], "—")
        rows.append([str(i), r["term"], str(r["freq"]), f"{r['c_value']:.2f}",
                     f"{r['weirdness']:.1f}", f"{r['termhood']:.2f}", v])
        if v.startswith("valid"):
            row_colors.append("#EAF6F0")
        elif v.startswith("partial"):
            row_colors.append("#FDF3E3")
        else:
            row_colors.append("#FBEAE9")

    render_table(cols, rows,
                 "Table 3 — Top ten multi-word candidates from the wind-energy use case\n"
                 "Seven of ten are valid terms; two show boundary errors; one is spurious",
                 "table3_extracted_terms.png",
                 col_chars=[5, 20, 5, 8, 10, 9, 44],
                 fontsize=9.2, row_colors=row_colors)


# =============================================================================
# TABLE 5 — Method families and when to use them
# =============================================================================
def table5_decision_guide():
    cols = ["Method family", "Representative system", "Labelled data",
            "Reported F1 (ACTER)", "Use when"]
    rows = [
        ["Unsupervised statistical", "C-value, weirdness (PyATE)", "None",
         "≈15–39", "No annotation exists; interpretability required"],
        ["Feature-engineered supervised", "HAMLET (152 features)", "In-domain, large",
         "54.2–66.1", "Annotated data exists; features are auditable"],
        ["Transformer token classification", "XLM-R token classifier", "In-domain, large",
         "58.3–69.6", "Best accuracy and annotation is available"],
        ["Cross-lingual zero-shot", "XLM-R, zero-shot transfer", "Other-language only",
         "58.3–69.8", "Target language has no annotated data"],
        ["LLM in-context learning", "GPT-3.5 / Llama-3.1 few-shot", "A handful of examples",
         "≈60 (cross-domain)", "Few-shot regime, or a new domain with no data"],
    ]
    render_table(cols, rows,
                 "Table 5 — Choosing an ATE method\n"
                 "F1 figures are as reported on the ACTER benchmark by "
                 "Tran et al. (2023) and Chun et al. (2025)",
                 "table5_decision_guide.png",
                 col_chars=[20, 25, 15, 14, 35], fontsize=9.2)


# =============================================================================
def main():
    print("Generating ATE article figures...")
    # formulas, in the order they appear in the article
    formula_pos_patterns()   # formula 1 - section 3
    formula_cvalue()         # formula 2 - section 4.2
    formula_cvalue_terms()   # formula 3 - section 4.2
    formula_ncvalue()        # formula 4 - section 4.5
    formula_weirdness()      # formula 5 - section 5
    formula_termhood()       # formula 6 - section 5
    formula_metrics()        # formula 7 - section 8
    formula_tor()            # formula 8 - section 11.3
    # figures and tables, in the order they appear in the article
    table1_keywords_vs_terms()      # Table 1  - section 1
    table3_annotation_categories()  # Table 2  - section 1.1
    fig1_pipeline()                 # Figure 1 - section 2
    fig2_frequency_vs_cvalue()      # Figure 2 - section 4.4
    fig3_termhood_plane()           # Figure 3 - section 5
    table4_extracted_terms()        # Table 3  - section 6.2
    fig7_threshold_and_length()     # Figure 4 - section 7
    table2_acter()                  # Table 4  - section 8
    fig5_literature_statistics()    # Figure 5 - section 9
    fig4_acter_f1()                 # Figure 6 - section 10
    fig6_llm_vs_plm()               # Figure 7 - section 11.2
    table5_decision_guide()         # Table 5  - section 12
    print("Done.")


if __name__ == "__main__":
    main()

