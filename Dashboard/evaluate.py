"""
Evaluation logic for ABA Mining Task 1.

Loads gold annotations from the Dataset/Task1 CSV and computes metrics
(topic F1, span verbatim-match F1, sentiment accuracy) against model JSONL outputs.

Also provides schema-agnostic output statistics that work without gold labels
(valid rate, null rate, avg spans per topic, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TOPICS = [
    "Room", "Staff", "Location", "Food", "Price",
    "Facility", "Check-in", "Check-out", "Off", "Booking-issue", "Taxi-issue",
]

GOLD_CSV = Path(__file__).resolve().parents[1] / "Dataset" / "Task1" / \
    "[Annotation work] Original ABA Dataset for Version 2 (25-07-2025 Nonny Version).xlsx - A-1. hotel in Larnaca-Cyprus.csv"


# ── Gold data loading ────────────────────────────────────────────────────────

def load_gold(gold_csv: Path = GOLD_CSV) -> dict[str, list[dict[str, str]]]:
    """Return {review_id: [{"topic": ..., "text": ..., "label": ...}, ...]}"""
    df = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
    gold: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        rid = str(row.get("ID", "")).strip()
        topic = str(row.get("Topic", "")).strip()
        text = str(row.get("Selected Content", "")).strip()
        label_raw = str(row.get("Pos/Neg", "")).strip()
        # Normalise label
        label = None
        if label_raw.lower() in ("positive", "pos"):
            label = "Positive"
        elif label_raw.lower() in ("negative", "neg"):
            label = "Negative"
        if not rid or not topic or not text:
            continue
        gold.setdefault(rid, []).append({"topic": topic, "text": text, "label": label})
    return gold


# ── Output loading ───────────────────────────────────────────────────────────

def load_outputs(jsonl_path: Path) -> list[dict[str, Any]]:
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Schema-agnostic output statistics ───────────────────────────────────────

def output_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics that don't require gold labels."""
    total = len(rows)
    if total == 0:
        return {"total": 0, "valid": 0, "valid_pct": 0.0, "parse_fail": 0}

    valid = sum(1 for r in rows if r.get("valid"))
    parse_fail = sum(1 for r in rows if r.get("errors") and any("json_parse_error" in e for e in r["errors"]))
    schema_errors = sum(1 for r in rows if r.get("errors") and not r.get("valid") and not any("json_parse_error" in e for e in r.get("errors", [])))

    # Per-schema stats for valid rows
    output_schema = rows[0].get("output_schema", "full") if rows else "full"
    avg_topics_found = 0.0
    avg_spans_per_review = 0.0
    avg_null_topics = 0.0
    contrastive_reviews = 0  # reviews with at least one topic having both pos+neg

    valid_rows = [r for r in rows if r.get("valid") and r.get("parsed")]

    if valid_rows:
        topics_found_list = []
        spans_list = []
        null_list = []
        for r in valid_rows:
            parsed = r["parsed"]
            topics_dict = parsed.get("Topics", {})

            if output_schema == "topic_only":
                found = sum(1 for v in topics_dict.values() if v is True)
                null_count = sum(1 for v in topics_dict.values() if v is False)
                topics_found_list.append(found)
                null_list.append(null_count)

            elif output_schema in ("span_only", "full"):
                found, spans, nulls, contrast = 0, 0, 0, 0
                for topic, items in topics_dict.items():
                    if not isinstance(items, list):
                        continue
                    non_null = [it for it in items if it.get("text")]
                    if non_null:
                        found += 1
                        spans += len(non_null)
                    else:
                        nulls += 1
                    # Contrastive: same topic has both Positive and Negative
                    if output_schema == "full":
                        labels = {it.get("label") for it in non_null if it.get("label")}
                        if "Positive" in labels and "Negative" in labels:
                            contrast += 1
                topics_found_list.append(found)
                spans_list.append(spans)
                null_list.append(nulls)
                if contrast > 0:
                    contrastive_reviews += 1

            elif output_schema == "sentiment_only":
                found = sum(1 for v in topics_dict.values() if v is not None)
                null_count = sum(1 for v in topics_dict.values() if v is None)
                topics_found_list.append(found)
                null_list.append(null_count)

        avg_topics_found = sum(topics_found_list) / len(topics_found_list) if topics_found_list else 0.0
        avg_spans_per_review = sum(spans_list) / len(spans_list) if spans_list else 0.0
        avg_null_topics = sum(null_list) / len(null_list) if null_list else 0.0

    return {
        "total": total,
        "valid": valid,
        "valid_pct": round(valid / total * 100, 1),
        "parse_fail": parse_fail,
        "schema_errors": schema_errors,
        "avg_topics_found": round(avg_topics_found, 2),
        "avg_spans_per_review": round(avg_spans_per_review, 2),
        "avg_null_topics": round(avg_null_topics, 2),
        "contrastive_reviews": contrastive_reviews,
        "output_schema": output_schema,
    }


# ── Gold-based evaluation ────────────────────────────────────────────────────

def eval_topic_f1(rows: list[dict], gold: dict[str, list[dict]]) -> dict[str, float]:
    """
    Subtask 1.1: Topic identification F1.
    Pred: topic is 'present' if model put any non-null item under it.
    Gold: topic is 'present' if the gold has at least one annotation for it.
    """
    tp = fp = fn = tn = 0
    for r in rows:
        if not r.get("valid") or not r.get("parsed"):
            continue
        rid = r["review_id"]
        gold_entries = gold.get(rid, [])
        gold_topics = {e["topic"] for e in gold_entries}

        topics_dict = r["parsed"].get("Topics", {})
        output_schema = r.get("output_schema", "full")

        for topic in TOPICS:
            # predicted presence
            val = topics_dict.get(topic)
            if output_schema == "topic_only":
                pred_present = val is True
            elif output_schema in ("span_only", "full"):
                pred_present = isinstance(val, list) and any(it.get("text") for it in val)
            elif output_schema == "sentiment_only":
                pred_present = val is not None
            else:
                pred_present = False

            gold_present = topic in gold_topics
            if pred_present and gold_present:
                tp += 1
            elif pred_present and not gold_present:
                fp += 1
            elif not pred_present and gold_present:
                fn += 1
            else:
                tn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r_val, 4), "f1": round(f1, 4), "accuracy": round(acc, 4)}


def eval_span_f1(rows: list[dict], gold: dict[str, list[dict]]) -> dict[str, float]:
    """
    Subtask 1.2: Span extraction F1 using exact verbatim substring match.
    A predicted span is a TP if it appears in the gold selected-content list for that topic.
    """
    tp = fp = fn = 0
    for r in rows:
        if not r.get("valid") or not r.get("parsed"):
            continue
        rid = r["review_id"]
        gold_entries = gold.get(rid, [])

        topics_dict = r["parsed"].get("Topics", {})
        output_schema = r.get("output_schema", "full")

        for topic in TOPICS:
            gold_texts = {e["text"].strip() for e in gold_entries if e["topic"] == topic and e["text"]}

            val = topics_dict.get(topic, [])
            if output_schema == "topic_only" or output_schema == "sentiment_only":
                continue  # no spans to compare

            if not isinstance(val, list):
                continue
            pred_texts = [it.get("text", "").strip() for it in val if it.get("text")]

            matched_gold = set()
            for pred in pred_texts:
                # Exact or substring match
                hit = pred in gold_texts or any(pred in g or g in pred for g in gold_texts)
                if hit:
                    tp += 1
                    matched_gold.add(pred)
                else:
                    fp += 1
            fn += len(gold_texts - matched_gold)

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r_val, 4), "f1": round(f1, 4)}


def eval_sentiment_f1(rows: list[dict], gold: dict[str, list[dict]]) -> dict[str, float]:
    """
    Subtask 1.3: Sentiment accuracy per (review, topic) pair.
    Only counts pairs where gold has an annotation (ignores null predictions on absent gold).
    """
    tp = fp = fn = tn = 0
    for r in rows:
        if not r.get("valid") or not r.get("parsed"):
            continue
        rid = r["review_id"]
        gold_entries = gold.get(rid, [])
        topics_dict = r["parsed"].get("Topics", {})
        output_schema = r.get("output_schema", "full")

        for topic in TOPICS:
            gold_labels = [e["label"] for e in gold_entries if e["topic"] == topic and e["label"]]
            if not gold_labels:
                continue  # skip topics not in gold
            # Dominant gold label (most common)
            gold_label = max(set(gold_labels), key=gold_labels.count)

            val = topics_dict.get(topic)
            if output_schema == "sentiment_only":
                pred_label = val
            elif output_schema == "full":
                if isinstance(val, list):
                    labels = [it.get("label") for it in val if it.get("label")]
                    pred_label = labels[0] if labels else None
                else:
                    pred_label = None
            else:
                continue

            if pred_label == gold_label:
                tp += 1
            elif pred_label is None:
                fn += 1
            else:
                fp += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
    acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r_val, 4), "f1": round(f1, 4), "accuracy": round(acc, 4)}


def evaluate_file(jsonl_path: Path, gold: dict | None = None) -> dict[str, Any]:
    """Full evaluation of a single JSONL output file."""
    rows = load_outputs(jsonl_path)
    stats = output_stats(rows)
    result: dict[str, Any] = {"file": str(jsonl_path), "stats": stats}

    if gold is None:
        try:
            gold = load_gold()
        except Exception:
            gold = {}

    output_schema = rows[0].get("output_schema", "full") if rows else "full"

    if output_schema in ("topic_only", "span_only", "full"):
        result["topic_f1"] = eval_topic_f1(rows, gold)
    if output_schema in ("span_only", "full"):
        result["span_f1"] = eval_span_f1(rows, gold)
    if output_schema in ("sentiment_only", "full"):
        result["sentiment_f1"] = eval_sentiment_f1(rows, gold)

    return result
