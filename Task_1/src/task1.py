from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator
from tqdm import tqdm

from .config import ModelConfig, PathsConfig, TopicsConfig
from .llm import LLMClient
from .prompts import load_prompt, render_prompt
from .utils import try_parse_json


@dataclass(frozen=True)
class Task1Instance:
    review_id: str
    review_text: str


def build_task1_schema(topics: list[str], output_schema: str = "full") -> dict[str, Any]:
    """Build JSON schema validator based on output schema type.

    Args:
        topics: List of topic names
        output_schema: One of "topic_only", "span_only", "sentiment_only", "full"
    """
    if output_schema == "topic_only":
        # {"Topics": {"Room": true/false, ...}}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "Topics": {
                    "type": "object",
                    "properties": {t: {"type": "boolean"} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["Topics"],
            "additionalProperties": False,
        }
    elif output_schema == "span_only":
        # {"Topics": {"Room": [{"text": "..."}], ...}}
        topic_obj = {
            "type": "object",
            "properties": {"text": {"type": ["string", "null"]}},
            "required": ["text"],
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "Topics": {
                    "type": "object",
                    "properties": {t: {"type": "array", "items": topic_obj, "minItems": 1} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["Topics"],
            "additionalProperties": False,
        }
    elif output_schema == "sentiment_only":
        # {"Topics": {"Room": "Positive"/"Negative"/null, ...}}
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "Topics": {
                    "type": "object",
                    "properties": {t: {"type": ["string", "null"]} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["Topics"],
            "additionalProperties": False,
        }
    else:  # full
        # {"Topics": {"Room": [{"text": "...", "label": "Positive"}], ...}}
        topic_obj = {
            "type": "object",
            "properties": {"text": {"type": ["string", "null"]}, "label": {"type": ["string", "null"]}},
            "required": ["text", "label"],
            "additionalProperties": False,
        }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "Topics": {
                    "type": "object",
                    "properties": {t: {"type": "array", "items": topic_obj, "minItems": 1} for t in topics},
                    "required": topics,
                    "additionalProperties": False,
                }
            },
            "required": ["Topics"],
            "additionalProperties": False,
        }

# checking if the above function is correct
def local_validate_task1(parsed: Any, *, topics: list[str], review_text: str, output_schema: str = "full") -> list[str]:
    errors: list[str] = []
    validator = Draft202012Validator(build_task1_schema(topics, output_schema))
    for e in validator.iter_errors(parsed):
        errors.append(e.message)

    if errors:
        return errors

    # Schema-specific semantic validation
    if output_schema == "topic_only":
        # {"Topics": {"Room": true/false, ...}}
        for t in topics:
            val = parsed["Topics"][t]
            if not isinstance(val, bool):
                errors.append(f"{t}: value must be boolean")
    elif output_schema == "span_only":
        # {"Topics": {"Room": [{"text": "..."}], ...}}
        for t in topics:
            items = parsed["Topics"][t]
            for it in items:
                text = it["text"]
                if text is not None and text not in review_text:
                    errors.append(f"{t}: text span not found verbatim in review_text")
    elif output_schema == "sentiment_only":
        # {"Topics": {"Room": "Positive"/"Negative"/null, ...}}
        for t in topics:
            val = parsed["Topics"][t]
            if val is not None and val not in ("Positive", "Negative"):
                errors.append(f"{t}: label must be 'Positive', 'Negative', or null")
    else:  # full
        # {"Topics": {"Room": [{"text": "...", "label": "Positive"}], ...}}
        for t in topics:
            items = parsed["Topics"][t]
            for it in items:
                text = it["text"]
                label = it["label"]
                if text is None:
                    if label is not None:
                        errors.append(f"{t}: label must be null when text is null")
                else:
                    if label not in ("Positive", "Negative"):
                        errors.append(f"{t}: label must be Positive/Negative when text is non-null")
                    if text not in review_text:
                        errors.append(f"{t}: text span not found verbatim in review_text")
    return errors


def load_task1_instances_from_input(paths: PathsConfig, limit_reviews: int | None = None, offset_reviews: int = 0) -> list[Task1Instance]:
    """Load reviews from the input CSV (A-1 dataset)."""
    df = pd.read_csv(paths.input_csv, dtype=str, keep_default_na=False)
    id_col = "Column1" if "Column1" in df.columns else "ID"
    if id_col not in df.columns:
        raise RuntimeError(f"Expected 'Column1' or 'ID' column in input_csv, found: {list(df.columns)}")

    grouped = df.groupby(id_col, sort=False, dropna=False)
    instances: list[Task1Instance] = []
    skipped = 0
    for review_id, g in grouped:
        if skipped < offset_reviews:
            skipped += 1
            continue
        pos = (g["PositiveReview"].iloc[0] if "PositiveReview" in g.columns else "") or ""
        neg = (g["NegativeReview"].iloc[0] if "NegativeReview" in g.columns else "") or ""
        review_text = f'PositiveReview — "{pos.strip()}"\nNegativeReview — "{neg.strip()}"'
        instances.append(Task1Instance(review_id=str(review_id), review_text=review_text))

        if limit_reviews is not None and len(instances) >= limit_reviews:
            break

    return instances


def _transform_to_schema(parsed: dict[str, Any], output_schema: str, topics: list[str]) -> dict[str, Any]:
    """Transform model output from full format to the target output schema."""
    if output_schema == "full":
        return parsed  # No transformation needed

    topics_dict = parsed.get("Topics", {})
    new_topics = {}

    if output_schema == "topic_only":
        # {"Topics": {"Room": true/false, ...}}
        for topic in topics:
            spans = topics_dict.get(topic, [])
            if isinstance(spans, list) and len(spans) > 0:
                # Topic is present if there are non-null spans
                new_topics[topic] = any(s.get("text") is not None for s in spans)
            else:
                new_topics[topic] = False
        return {"Topics": new_topics}

    elif output_schema == "span_only":
        # {"Topics": {"Room": [{"text": "..."}], ...}}
        for topic in topics:
            spans = topics_dict.get(topic, [])
            if isinstance(spans, list):
                new_spans = [{"text": s.get("text")} for s in spans]
                new_topics[topic] = new_spans
            else:
                new_topics[topic] = [{"text": None}]
        return {"Topics": new_topics}

    elif output_schema == "sentiment_only":
        # {"Topics": {"Room": "Positive"/"Negative"/null, ...}}
        for topic in topics:
            spans = topics_dict.get(topic, [])
            if isinstance(spans, list) and len(spans) > 0:
                # Take the first non-null sentiment (or null if all are null)
                labels = [s.get("label") for s in spans if s.get("text") is not None]
                new_topics[topic] = labels[0] if labels else None
            else:
                new_topics[topic] = None
        return {"Topics": new_topics}

    return parsed


def run_task1(
    *,
    repo_root: Path,
    client: LLMClient,
    topics_cfg: TopicsConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    limit_reviews: int = 20,
    offset_reviews: int = 0,
    prompt_path: str | None = "prompts/task1/Contrastive/generator_v1.txt",
    prompt_template: str | None = None,
    output_label: str | None = None,
    output_schema: str = "full",
    validator_path: str = "prompts/task1/Contrastive/validator_v1.txt",
    max_retries: int = 2,
    output_subdir: str | None = None,
) -> Path:
    out_dir = paths_cfg.task1_dir
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_cfg.task1_model.replace(":", "_").replace("/", "_")
    prompt_label = output_label or (Path(prompt_path).stem if prompt_path else "custom")
    out_path = out_dir / f"task1_{model_name}_{topics_cfg.active_schema}_{prompt_label}_n{limit_reviews}.jsonl"

    if prompt_template is None:
        if prompt_path is None:
            raise ValueError("Provide either prompt_path or prompt_template.")
        gen_template = load_prompt(repo_root, prompt_path)
    else:
        gen_template = prompt_template
    val_template = load_prompt(repo_root, validator_path)
    topics_str = ", ".join(topics_cfg.topics)
    topic_count = str(len(topics_cfg.topics))

    instances = load_task1_instances_from_input(paths_cfg, limit_reviews=limit_reviews, offset_reviews=offset_reviews)

    def _generate(inst: Task1Instance) -> tuple[str, bool, Any, list[str]]:
        """Run the generator and return (raw_text, ok, parsed, errors)."""
        prompt = render_prompt(
            gen_template,
            TOPICS=topics_str,
            TOPIC_COUNT=topic_count,
            REVIEW_TEXT=inst.review_text,
        )
        resp = client.complete(
            model=model_cfg.task1_model,
            prompt=prompt,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_output_tokens=model_cfg.max_output_tokens,
        )
        ok, parsed, err = try_parse_json(resp.text)
        errors: list[str] = []
        if not ok:
            errors = [f"json_parse_error: {err}"]
        else:
            errors = local_validate_task1(parsed, topics=topics_cfg.topics, review_text=inst.review_text, output_schema=output_schema)
        return resp.text, ok, parsed, errors

    def _validate_and_fix(inst: Task1Instance, candidate_json: str, errors: list[str]) -> tuple[str, bool, Any, list[str]]:
        """Send the candidate + errors to the validator LLM for correction.
        Only used for 'full' schema — the validator prompt expects {text, label} format."""
        if output_schema != "full":
            return candidate_json, False, None, errors  # skip validator for non-full schemas
        prompt = render_prompt(
            val_template,
            TOPICS=topics_str,
            REVIEW_TEXT=inst.review_text,
            CANDIDATE_JSON=candidate_json,
            ERRORS="; ".join(errors),
        )
        resp = client.complete(
            model=model_cfg.task1_model,
            prompt=prompt,
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            max_output_tokens=model_cfg.max_output_tokens,
        )
        ok, parsed, err = try_parse_json(resp.text)
        new_errors: list[str] = []
        if not ok:
            new_errors = [f"json_parse_error: {err}"]
        else:
            new_errors = local_validate_task1(parsed, topics=topics_cfg.topics, review_text=inst.review_text, output_schema=output_schema)
        return resp.text, ok, parsed, new_errors

    def _process_one(inst: Task1Instance) -> dict[str, Any]:
        raw, ok, parsed, errors = _generate(inst)
        attempt = 0

        # Cascading Validation Loop
        while errors and attempt < max_retries:
            attempt += 1
            candidate_str = raw if not ok else json.dumps(parsed, ensure_ascii=False)
            val_raw, val_ok, val_parsed, val_errors = _validate_and_fix(inst, candidate_str, errors)
            # Accept the fix only if it's an improvement (fewer or no errors)
            if val_ok and len(val_errors) < len(errors):
                raw, ok, parsed, errors = val_raw, val_ok, val_parsed, val_errors
            elif val_ok and len(val_errors) == 0:
                raw, ok, parsed, errors = val_raw, val_ok, val_parsed, val_errors
                break
            else:
                # Validator didn't help
                break

        # If we have a valid parse, compact the raw_output to remove newlines/indentation
        compact_raw = json.dumps(parsed, ensure_ascii=False) if ok and parsed is not None else raw

        return {
            "review_id": inst.review_id,
            "schema": topics_cfg.active_schema,
            "output_schema": output_schema,
            "prompt": prompt_path,
            "raw_output": compact_raw,
            "parsed": parsed if ok else None,
            "valid": len(errors) == 0,
            "errors": errors,
            "retries": attempt,
        }

    # Parallelism for local serving throughput
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict[str, Any]] = [None] * len(instances) 
    with ThreadPoolExecutor(max_workers=model_cfg.num_workers) as ex:
        fut_to_idx = {ex.submit(_process_one, inst): i for i, inst in enumerate(instances)}
        for fut in tqdm(as_completed(fut_to_idx), total=len(instances), desc="Task1"):
            i = fut_to_idx[fut]
            results[i] = fut.result()

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = out_path.with_suffix(".csv")
    _write_readable_csv(csv_path, results, instances, output_schema=output_schema)

    return out_path


def _write_readable_csv(csv_path: Path, results: list[dict[str, Any]], instances: list[Task1Instance], output_schema: str = "full") -> None:
    """Write a human-readable CSV with output appropriate to the schema type."""
    import csv as csv_mod

    inst_map = {inst.review_id: inst.review_text for inst in instances}

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv_mod.writer(f)

        if output_schema == "topic_only":
            writer.writerow(["Review ID", "Valid", "Topics Present", "Errors"])
            for r in results:
                review_id = r["review_id"]
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "Topics" in parsed:
                    present = [t for t, v in parsed["Topics"].items() if v]
                    topics_str = ", ".join(present) if present else "(none)"
                    writer.writerow([review_id, valid, topics_str, errors])
                else:
                    writer.writerow([review_id, valid, "(parse failed)", errors])
        elif output_schema == "span_only":
            writer.writerow(["Review ID", "Valid", "Topic", "Text Span", "Errors"])
            for r in results:
                review_id = r["review_id"]
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "Topics" in parsed:
                    found_any = False
                    for topic, spans in parsed["Topics"].items():
                        if not isinstance(spans, list):
                            continue
                        for span in spans:
                            if span is None or not isinstance(span, dict):
                                continue
                            text = span.get("text")
                            if not text or text == "null":
                                continue
                            found_any = True
                            writer.writerow([review_id, valid, topic, text, ""])
                    if not found_any:
                        writer.writerow([review_id, valid, "(no spans found)", "", errors])
                else:
                    writer.writerow([review_id, valid, "(parse failed)", "", errors])
        elif output_schema == "sentiment_only":
            writer.writerow(["Review ID", "Valid", "Topic", "Sentiment", "Errors"])
            for r in results:
                review_id = r["review_id"]
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "Topics" in parsed:
                    found_any = False
                    for topic, label in parsed["Topics"].items():
                        if label is not None:
                            found_any = True
                            writer.writerow([review_id, valid, topic, label, ""])
                    if not found_any:
                        writer.writerow([review_id, valid, "(no sentiments found)", "", errors])
                else:
                    writer.writerow([review_id, valid, "(parse failed)", "", errors])
        else:  # full
            writer.writerow(["Review ID", "Valid", "Topic", "Text Span", "Sentiment", "Errors"])
            for r in results:
                review_id = r["review_id"]
                valid = r["valid"]
                errors = "; ".join(r.get("errors", []))
                parsed = r.get("parsed")
                if parsed and "Topics" in parsed:
                    found_any = False
                    for topic, spans in parsed["Topics"].items():
                        if not isinstance(spans, list):
                            continue
                        for span in spans:
                            if span is None or not isinstance(span, dict):
                                continue
                            text = span.get("text")
                            label = span.get("label")
                            if not text or text == "null":
                                continue
                            found_any = True
                            writer.writerow([review_id, valid, topic, text, label or "", ""])
                    if not found_any:
                        writer.writerow([review_id, valid, "(no topics found)", "", "", errors])
                else:
                    writer.writerow([review_id, valid, "(parse failed)", "", "", errors])

