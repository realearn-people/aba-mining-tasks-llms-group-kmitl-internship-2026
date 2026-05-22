"""
Scatter plot to help choose a cosine-similarity threshold for subtask 1.2.

Two panels:
  Top    – scatter: X = similarity score, Y = row index (one dot per span)
           Vertical dashed lines at thresholds 0.3–0.8; dots coloured by
           the highest threshold they pass.
  Bottom – histogram of similarity scores with the same threshold lines
           + a cumulative pass-rate curve on the right axis.

Usage:
    python Task_1/plot_threshold.py                  # all output CSVs combined
    python Task_1/plot_threshold.py --single         # llama4_scout n20 only
    python Task_1/plot_threshold.py --pred <path>    # specific CSV
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves to file without needing a display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "Task_1/outputs/task1"
DEFAULT_PRED = ROOT / "Task_1/outputs/task1/llama4_scout/modular/subtask1_2/task1_llama4_scout_extended11_subtask1_2_n20.csv"
REVIEWS_CSV = ROOT / "Dataset/Original ABA Dataset for Version 2 [Oct 23, 2025], Senior Project, MUICT - Sheet2_.csv"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
COLORS = ["#e41a1c", "#ff7f00", "#e6b800", "#4daf4a", "#377eb8", "#984ea3"]
GRAY = "#bbbbbb"


# ── helpers ────────────────────────────────────────────────────────────────────
def load_reviews(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    reviews: dict[str, str] = {}
    for _, row in df.iterrows():
        rid = str(row.get("ID", "")).strip()
        if not rid:
            continue
        pos = row.get("PositiveReview", "").strip()
        neg = row.get("NegativeReview", "").strip()
        reviews[rid] = (pos + " " + neg).strip()
    return reviews


_SKIP = {"", "(no spans found)", "(parse failed)"}

def load_all_predictions(outputs_dir: Path) -> pd.DataFrame:
    """Collect every CSV under outputs_dir that has a 'Text Span' column."""
    frames = []
    for path in sorted(outputs_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            df.columns = df.columns.str.strip()
            if "Text Span" not in df.columns or "Review ID" not in df.columns:
                continue
            df = df[~df["Text Span"].isin(_SKIP)].copy()
            if df.empty:
                continue
            df["_source"] = path.stem
            frames.append(df[["Review ID", "Topic", "Text Span", "_source"]])
        except Exception:
            continue

    if not frames:
        raise RuntimeError(f"No usable CSV files found under {outputs_dir}")

    combined = pd.concat(frames, ignore_index=True)
    # Deduplicate identical (review_id, span) pairs across runs
    combined = combined.drop_duplicates(subset=["Review ID", "Text Span"])
    return combined.reset_index(drop=True)


def load_single_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    df = df[~df["Text Span"].isin(_SKIP)].copy()
    df["_source"] = path.stem
    return df[["Review ID", "Topic", "Text Span", "_source"]].reset_index(drop=True)


def compute_scores(preds: pd.DataFrame, reviews: dict[str, str]) -> np.ndarray:
    """
    For each span, compute TF-IDF char-ngram cosine similarity against
    its source review.  Group by review_id for efficiency.
    """
    scores = np.zeros(len(preds), dtype=float)

    for rid, group in preds.groupby("Review ID"):
        review_text = reviews.get(str(rid).strip(), "")
        spans = group["Text Span"].tolist()
        if not review_text:
            # No review found: leave scores as 0
            continue
        try:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
            texts = [review_text] + spans
            tfidf = vec.fit_transform(texts)
            sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
            scores[group.index] = sims
        except Exception:
            pass  # leave as 0

    return scores


def threshold_color(s: float) -> str:
    for thresh, color in zip(reversed(THRESHOLDS), reversed(COLORS)):
        if s >= thresh:
            return color
    return GRAY


# ── main ───────────────────────────────────────────────────────────────────────
def main(pred_path: Path | None, use_all: bool) -> None:
    print("Loading reviews…")
    reviews = load_reviews(REVIEWS_CSV)
    print(f"  {len(reviews)} reviews")

    if use_all:
        print(f"Loading ALL prediction CSVs under {OUTPUTS_DIR}…")
        preds = load_all_predictions(OUTPUTS_DIR)
    else:
        print(f"Loading {pred_path}…")
        preds = load_single_predictions(pred_path)

    print(f"  {len(preds)} spans (after dedup)")

    print("Computing similarity scores…")
    scores = compute_scores(preds, reviews)
    preds = preds.copy()
    preds["score"] = scores
    preds["color"] = [threshold_color(s) for s in scores]

    # ── figure ─────────────────────────────────────────────────────────────────
    n = len(preds)
    scatter_h = max(8, min(60, n * 0.003))   # sensible height cap
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, scatter_h + 4),
        gridspec_kw={"height_ratios": [scatter_h, 4]},
    )
    fig.suptitle("Subtask 1.2 — Similarity score distribution\n"
                 "(choose threshold where most spans cluster to the right)",
                 fontsize=13, fontweight="bold", y=1.0)

    # ── top: scatter ───────────────────────────────────────────────────────────
    y_vals = np.arange(n)
    ax_top.scatter(
        preds["score"], y_vals,
        c=preds["color"],
        s=6, alpha=1.0,
        linewidths=0, rasterized=True,
    )

    for thresh, color in zip(THRESHOLDS, COLORS):
        ax_top.axvline(thresh, color=color, linestyle="--", linewidth=1.4, alpha=0.9)
        n_pass = (preds["score"] >= thresh).sum()
        rate = n_pass / n * 100
        ax_top.text(
            thresh + 0.008, n * 0.99,
            f"{rate:.0f}%",
            color=color, fontsize=9, va="top", fontweight="bold",
        )

    ax_top.set_xlabel("Cosine similarity (TF-IDF char n-gram, 2-3)", fontsize=11)
    ax_top.set_ylabel(f"Row index  (N = {n:,})", fontsize=11)
    ax_top.set_xlim(-0.02, 1.02)
    ax_top.set_ylim(-n * 0.01, n * 1.01)
    ax_top.grid(axis="x", linestyle=":", alpha=0.4)

    legend_handles = (
        [Line2D([0], [0], color=c, linestyle="--", linewidth=1.5, label=f"≥ {t}") for t, c in zip(THRESHOLDS, COLORS)]
        + [mpatches.Patch(color=GRAY, label=f"< {THRESHOLDS[0]}")]
    )
    ax_top.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.85)

    # ── bottom: histogram + cumulative line ────────────────────────────────────
    bins = np.linspace(0, 1, 51)
    counts, edges = np.histogram(scores, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    bar_colors = [threshold_color(c) for c in bin_centers]
    ax_bot.bar(bin_centers, counts, width=(bins[1] - bins[0]) * 0.9,
               color=bar_colors, alpha=0.85, linewidth=0)

    for thresh, color in zip(THRESHOLDS, COLORS):
        ax_bot.axvline(thresh, color=color, linestyle="--", linewidth=1.4, alpha=0.9)

    ax_bot.set_xlabel("Cosine similarity score", fontsize=11)
    ax_bot.set_ylabel("Number of spans", fontsize=11)
    ax_bot.set_xlim(-0.02, 1.02)
    ax_bot.grid(axis="y", linestyle=":", alpha=0.4)

    ax2 = ax_bot.twinx()
    cum_pct = np.array([(scores >= t).sum() / n * 100 for t in np.linspace(0, 1, 200)])
    ax2.plot(np.linspace(0, 1, 200), cum_pct, color="black", linewidth=1.8, label="pass rate %")
    ax2.set_ylabel("Cumulative pass rate (%)", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    label = "all" if use_all else pred_path.stem
    out_path = ROOT / f"Task_1/outputs/threshold_scatter_{label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\nPass-rate summary  (N = {n:,})")
    print(f"  {'Threshold':<12} {'Pass':>6} {'%':>7}")
    print("  " + "-" * 28)
    for thresh in THRESHOLDS:
        n_pass = (scores >= thresh).sum()
        print(f"  >= {thresh:<9} {n_pass:>6,}  {n_pass/n*100:>6.1f}%")

    print(f"Open the image: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--pred", default=None, help="Path to a specific subtask1_2 CSV")
    grp.add_argument("--single", action="store_true",
                     help=f"Use only the default single file: {DEFAULT_PRED.name}")
    args = parser.parse_args()

    if args.pred:
        main(Path(args.pred), use_all=False)
    elif args.single:
        main(DEFAULT_PRED, use_all=False)
    else:
        main(None, use_all=True)
