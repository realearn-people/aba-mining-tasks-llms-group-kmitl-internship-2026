"""Task 2 evaluation — Phase 5 skeleton.

Gold source: Dataset/Task2/Task 2 Verification … - Task 2 Verification.csv
  - Filter Label == 1
  - Group by (review_id, topic) → set of accepted support bodies
  - Evaluate ONLY on the ~1,897 verified (ID, Topic) pairs.

Metrics (§5.1):
  micro-precision  — per-instance precision   (one value per row)
  micro-recall     — per-instance recall       (one value per row)
  average-precision — mean of micro-precisions
  average-recall    — mean of micro-recalls

Target from paper: avg micro-precision ≈ 0.88, avg micro-recall ≈ 0.94
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from Task_2.src.matcher import precision as mp, recall as mr

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent.parent
GOLD_CSV = (
    _REPO_ROOT
    / "Dataset"
    / "Task2"
    / "Task 2 Verification by Nonraphan and Yotsawan - Task 2 Verification.csv"
)


# ── Data types ────────────────────────────────────────────────────────────────
GoldIndex = dict[tuple[int, str], set[str]]   # (review_id, topic) → accepted bodies


class EvalRow(NamedTuple):
    review_id: int
    topic: str
    gold: set[str]
    preds: list[str]
    micro_precision: float
    micro_recall: float


# ── Gold loading ──────────────────────────────────────────────────────────────

def load_gold(csv_path: Path = GOLD_CSV) -> GoldIndex:
    """Load the verification CSV and return the gold index.

    Filters to Label == 1 rows, groups by (review_id, topic).
    Prints a sanity-check summary to stdout.
    """
    df = pd.read_csv(csv_path, dtype=str)

    # Drop header-artefact rows (non-numeric IDs)
    df = df[pd.to_numeric(df["ID"], errors="coerce").notna()].copy()
    df["ID"] = df["ID"].astype(int)

    # Normalise label column: "1" / 1.0 / "1.0" → 1
    df["_label_num"] = pd.to_numeric(df["Label"], errors="coerce")
    df_pos = df[df["_label_num"] == 1].copy()

    topic_col = "2. Topic-relevant Text"
    body_col = "Body"

    # Group by (review_id, topic) → set of accepted body strings
    gold: GoldIndex = {}
    for (rid, topic), grp in df_pos.groupby(["ID", topic_col]):
        bodies = {
            b.strip().lower()
            for b in grp[body_col].dropna()
            if str(b).strip()
        }
        if bodies:
            gold[(int(rid), str(topic))] = bodies

    n_pairs = len(gold)
    n_bodies = sum(len(v) for v in gold.values())

    print(
        f"[evaluate] Gold loaded from: {csv_path.name}\n"
        f"  Verified (ID, Topic) pairs : {n_pairs:,}   (expected ~1,897)\n"
        f"  Total accepted bodies      : {n_bodies:,}   (expected ~4,141)"
    )
    return gold


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    predictions: list[dict],
    gold: GoldIndex,
) -> list[EvalRow]:
    """Compute micro-precision and micro-recall for each verified pair.

    predictions: list of pipeline output dicts with keys:
        review_id, topic, supports (list[str])
    Returns only rows whose (review_id, topic) key is in gold.
    """
    rows: list[EvalRow] = []
    for pred in predictions:
        key = (int(pred["review_id"]), str(pred["topic"]))
        if key not in gold:
            continue
        gold_set = gold[key]
        preds_list = [s.lower().strip() for s in pred.get("supports", [])]
        rows.append(EvalRow(
            review_id=key[0],
            topic=key[1],
            gold=gold_set,
            preds=preds_list,
            micro_precision=mp(preds_list, gold_set),
            micro_recall=mr(preds_list, gold_set),
        ))
    return rows


def summarise(rows: list[EvalRow]) -> dict:
    if not rows:
        return {}
    avg_p = sum(r.micro_precision for r in rows) / len(rows)
    avg_r = sum(r.micro_recall    for r in rows) / len(rows)
    return {
        "n_evaluated": len(rows),
        "avg_micro_precision": round(avg_p, 4),
        "avg_micro_recall":    round(avg_r, 4),
    }


# ── CLI entry-point (Phase 5) ─────────────────────────────────────────────────
if __name__ == "__main__":
    gold = load_gold()
    print("Gold loaded successfully. Full evaluation requires predictions file.")
    print("Run: python -m Task_2.src.evaluate --predictions Task_2/outputs/task2_predictions.jsonl")
