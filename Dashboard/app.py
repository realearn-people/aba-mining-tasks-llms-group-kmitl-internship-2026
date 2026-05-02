"""
ABA Mining — Experiment Comparison Dashboard

Run from the Dashboard/ folder:
    streamlit run app.py

Or from project root:
    streamlit run Dashboard/app.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
TASK1_DIR = ROOT / "Task_1"
OUTPUTS_DIR = TASK1_DIR / "outputs" / "task1"
EXPERIMENTS_YAML = TASK1_DIR / "configs" / "experiments.yaml"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_gold, load_outputs, output_stats, eval_topic_f1, eval_span_f1, eval_sentiment_f1


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_experiments() -> dict:
    with EXPERIMENTS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("experiments", {})


@st.cache_data(show_spinner=False)
def get_gold() -> dict:
    try:
        return load_gold()
    except Exception as e:
        return {}


def discover_jsonl_files() -> list[Path]:
    """Find all JSONL files under outputs/task1/."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(OUTPUTS_DIR.rglob("*.jsonl"))


def exp_label_from_path(p: Path) -> str:
    """Extract experiment label from output file path."""
    # e.g. outputs/task1/llama4_scout/modular/combined/task1_llama4_scout_extended11_combined_n20.jsonl
    parts = p.relative_to(OUTPUTS_DIR).parts
    model = parts[0] if parts else "unknown"
    # strip the stem prefix to get the experiment label embedded in filename
    stem = p.stem  # e.g. task1_llama4_scout_extended11_combined_n20
    tokens = stem.split("_")
    # Find n<number> suffix
    n_idx = next((i for i, t in enumerate(tokens) if t.startswith("n") and t[1:].isdigit()), len(tokens))
    # Skip task1, model tokens, schema token — heuristic: take the token after the schema
    # Simpler: use folder name as experiment label
    if len(parts) >= 3:
        return f"{model} / {parts[1]} / {parts[2]}"
    return str(p.relative_to(OUTPUTS_DIR))


@st.cache_data(show_spinner=False)
def load_all_results(_files_key: str) -> pd.DataFrame:
    """Load statistics for all discovered JSONL files into a DataFrame."""
    files = discover_jsonl_files()
    gold = get_gold()
    records = []
    for f in files:
        rows = load_outputs(f)
        if not rows:
            continue
        stats = output_stats(rows)
        output_schema = stats.get("output_schema", "full")

        rec: dict = {
            "label": exp_label_from_path(f),
            "file": str(f.relative_to(ROOT)),
            "model": f.relative_to(OUTPUTS_DIR).parts[0] if f.relative_to(OUTPUTS_DIR).parts else "?",
            "mode": f.relative_to(OUTPUTS_DIR).parts[1] if len(f.relative_to(OUTPUTS_DIR).parts) > 1 else "?",
            "experiment": f.relative_to(OUTPUTS_DIR).parts[2] if len(f.relative_to(OUTPUTS_DIR).parts) > 2 else "?",
            "output_schema": output_schema,
            "n": stats["total"],
            "valid": stats["valid"],
            "valid_pct": stats["valid_pct"],
            "parse_fail": stats["parse_fail"],
            "schema_errors": stats["schema_errors"],
            "avg_topics_found": stats["avg_topics_found"],
            "avg_spans": stats["avg_spans_per_review"],
            "avg_null_topics": stats["avg_null_topics"],
            "contrastive_reviews": stats.get("contrastive_reviews", 0),
        }

        # Gold-based metrics
        if gold and output_schema in ("topic_only", "span_only", "full"):
            tf = eval_topic_f1(rows, gold)
            rec["topic_f1"] = tf["f1"]
            rec["topic_precision"] = tf["precision"]
            rec["topic_recall"] = tf["recall"]
        else:
            rec["topic_f1"] = rec["topic_precision"] = rec["topic_recall"] = None

        if gold and output_schema in ("span_only", "full"):
            sf = eval_span_f1(rows, gold)
            rec["span_f1"] = sf["f1"]
            rec["span_precision"] = sf["precision"]
            rec["span_recall"] = sf["recall"]
        else:
            rec["span_f1"] = rec["span_precision"] = rec["span_recall"] = None

        if gold and output_schema in ("sentiment_only", "full"):
            sentf = eval_sentiment_f1(rows, gold)
            rec["sent_f1"] = sentf["f1"]
            rec["sent_precision"] = sentf["precision"]
            rec["sent_recall"] = sentf["recall"]
        else:
            rec["sent_f1"] = rec["sent_precision"] = rec["sent_recall"] = None

        records.append(rec)

    return pd.DataFrame(records)


# ── Page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ABA Mining Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 ABA Mining — Experiment Dashboard")
st.caption("Research comparison across all rule/subtask combinations for Task 1")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Refresh results"):
    st.cache_data.clear()
    st.rerun()

# Discover files
all_files = discover_jsonl_files()
files_key = "|".join(str(f) for f in all_files)

if not all_files:
    st.warning("No output JSONL files found yet. Run experiments first.")
    st.info(f"Expected location: `{OUTPUTS_DIR}`")
    st.stop()

df = load_all_results(files_key)

# Sidebar filters
models_available = sorted(df["model"].unique().tolist())
selected_models = st.sidebar.multiselect("Filter by model", models_available, default=models_available)
schemas_available = sorted(df["output_schema"].unique().tolist())
selected_schemas = st.sidebar.multiselect("Filter by schema", schemas_available, default=schemas_available)

df_filtered = df[df["model"].isin(selected_models) & df["output_schema"].isin(selected_schemas)].copy()

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_metrics, tab_chart, tab_inspect, tab_run = st.tabs([
    "📋 Overview", "📊 Metrics", "📈 Charts", "🔍 Inspect Output", "▶️ Run Experiments"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("All Experiments Found")
    st.caption("One row per output file discovered.")

    display_cols = ["label", "model", "experiment", "output_schema", "n", "valid", "valid_pct",
                    "parse_fail", "schema_errors", "avg_topics_found", "avg_spans", "avg_null_topics"]
    display_df = df_filtered[display_cols].copy()
    display_df.columns = [
        "Label", "Model", "Experiment", "Schema", "N", "Valid",
        "Valid %", "Parse Fail", "Schema Err", "Avg Topics", "Avg Spans", "Avg Null Topics"
    ]

    # Color valid_pct column
    def highlight_valid(val):
        if isinstance(val, (int, float)):
            color = "#d4edda" if val >= 80 else "#fff3cd" if val >= 50 else "#f8d7da"
            return f"background-color: {color}"
        return ""

    styled = display_df.style.applymap(highlight_valid, subset=["Valid %"])
    st.dataframe(styled, use_container_width=True, height=450)

    # Summary cards
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total experiment files", len(df_filtered))
    c2.metric("Avg valid %", f"{df_filtered['valid_pct'].mean():.1f}%" if not df_filtered.empty else "—")
    c3.metric("Best valid %", f"{df_filtered['valid_pct'].max():.1f}%" if not df_filtered.empty else "—")
    c4.metric("Worst valid %", f"{df_filtered['valid_pct'].min():.1f}%" if not df_filtered.empty else "—")

    # Configured experiments (from yaml)
    st.divider()
    st.subheader("Configured Experiments (from experiments.yaml)")
    exps = get_experiments()
    exp_rows = []
    for name, cfg in exps.items():
        ran = df["experiment"].eq(name).any()
        exp_rows.append({
            "Experiment": name,
            "Rules": str(cfg.get("rules", [])),
            "Subtasks": str(cfg.get("subtasks", [])),
            "Schema": cfg.get("output_schema", "full"),
            "Status": "✅ ran" if ran else "⬜ not run",
            "Description": cfg.get("description", ""),
        })
    st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, height=550)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: Metrics (F1 comparison)
# ═══════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.subheader("Gold-based Evaluation Metrics")
    gold = get_gold()
    if not gold:
        st.warning("Gold annotation file not found or could not be loaded. Showing output stats only.")
    else:
        st.caption(f"Gold annotations loaded for {len(gold)} reviews.")

    metrics_cols = [
        "label", "experiment", "output_schema",
        "topic_f1", "topic_precision", "topic_recall",
        "span_f1", "span_precision", "span_recall",
        "sent_f1", "sent_precision", "sent_recall",
        "valid_pct",
    ]
    mdf = df_filtered[metrics_cols].copy()
    mdf.columns = [
        "Label", "Experiment", "Schema",
        "Topic F1", "Topic P", "Topic R",
        "Span F1", "Span P", "Span R",
        "Sent F1", "Sent P", "Sent R",
        "Valid %",
    ]

    def color_f1(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "color: #aaa"
        if isinstance(val, float):
            color = "#155724" if val >= 0.7 else "#856404" if val >= 0.4 else "#721c24"
            return f"color: {color}; font-weight: bold"
        return ""

    styled_m = mdf.style.applymap(color_f1, subset=["Topic F1", "Span F1", "Sent F1"])
    st.dataframe(styled_m, use_container_width=True, height=500)

    # Download
    csv_bytes = mdf.to_csv(index=False).encode()
    st.download_button("⬇️ Download metrics CSV", csv_bytes, "aba_metrics.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: Charts
# ═══════════════════════════════════════════════════════════════════════════
with tab_chart:
    st.subheader("Visual Comparison")

    if df_filtered.empty:
        st.info("No data to chart.")
    else:
        # Valid % bar chart
        st.markdown("#### Valid Output Rate (%)")
        fig_valid = px.bar(
            df_filtered.sort_values("valid_pct", ascending=False),
            x="label", y="valid_pct",
            color="model", barmode="group",
            labels={"valid_pct": "Valid %", "label": "Experiment"},
            height=350,
        )
        fig_valid.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_valid, use_container_width=True)

        # F1 comparison (for experiments that have gold metrics)
        f1_df = df_filtered.dropna(subset=["topic_f1", "span_f1", "sent_f1"], how="all").copy()
        if not f1_df.empty:
            st.markdown("#### F1 Score Comparison")
            chart_data = []
            for _, row in f1_df.iterrows():
                if row["topic_f1"] is not None and not pd.isna(row["topic_f1"]):
                    chart_data.append({"label": row["label"], "metric": "Topic F1", "value": row["topic_f1"]})
                if row["span_f1"] is not None and not pd.isna(row["span_f1"]):
                    chart_data.append({"label": row["label"], "metric": "Span F1", "value": row["span_f1"]})
                if row["sent_f1"] is not None and not pd.isna(row["sent_f1"]):
                    chart_data.append({"label": row["label"], "metric": "Sent F1", "value": row["sent_f1"]})
            if chart_data:
                fig_f1 = px.bar(
                    pd.DataFrame(chart_data).sort_values(["metric", "value"], ascending=[True, False]),
                    x="label", y="value", color="metric", barmode="group",
                    labels={"value": "F1 Score", "label": "Experiment", "metric": "Subtask"},
                    height=400,
                )
                fig_f1.update_layout(xaxis_tickangle=-45, yaxis_range=[0, 1])
                st.plotly_chart(fig_f1, use_container_width=True)

        # Avg topics found
        st.markdown("#### Avg Topics Found per Review")
        fig_topics = px.bar(
            df_filtered.sort_values("avg_topics_found", ascending=False),
            x="label", y="avg_topics_found",
            color="output_schema",
            labels={"avg_topics_found": "Avg Topics", "label": "Experiment"},
            height=320,
        )
        fig_topics.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_topics, use_container_width=True)

        # Radar chart: per-experiment subtask balance (for full-schema experiments only)
        full_df = df_filtered[df_filtered["output_schema"] == "full"].dropna(subset=["topic_f1", "span_f1", "sent_f1"])
        if len(full_df) >= 1:
            st.markdown("#### Radar: Subtask Balance (full schema experiments)")
            fig_radar = go.Figure()
            categories = ["Topic F1", "Span F1", "Sent F1", "Valid %"]
            for _, row in full_df.iterrows():
                vals = [
                    row["topic_f1"] or 0,
                    row["span_f1"] or 0,
                    row["sent_f1"] or 0,
                    (row["valid_pct"] or 0) / 100,
                ]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=row["label"],
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: Inspect individual outputs
# ═══════════════════════════════════════════════════════════════════════════
with tab_inspect:
    st.subheader("Inspect a Specific Experiment Output")

    file_options = {exp_label_from_path(f): f for f in discover_jsonl_files()}
    if not file_options:
        st.info("No output files found.")
    else:
        selected_label = st.selectbox("Select output file", list(file_options.keys()))
        selected_file = file_options[selected_label]
        rows = load_outputs(selected_file)

        stats = output_stats(rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total reviews", stats["total"])
        c2.metric("Valid", f"{stats['valid']} ({stats['valid_pct']}%)")
        c3.metric("Parse failures", stats["parse_fail"])
        c4.metric("Avg topics found", stats["avg_topics_found"])

        st.divider()
        filter_choice = st.radio("Show", ["All", "Valid only", "Invalid only"], horizontal=True)
        if filter_choice == "Valid only":
            display_rows = [r for r in rows if r.get("valid")]
        elif filter_choice == "Invalid only":
            display_rows = [r for r in rows if not r.get("valid")]
        else:
            display_rows = rows

        for i, r in enumerate(display_rows):
            status_icon = "✅" if r.get("valid") else "❌"
            with st.expander(f"{status_icon} Review {r['review_id']}  |  retries={r.get('retries', 0)}"):
                if r.get("errors"):
                    st.error("Errors: " + " | ".join(r["errors"][:3]))

                parsed = r.get("parsed")
                if parsed and "Topics" in parsed:
                    output_schema = r.get("output_schema", "full")
                    topic_rows = []
                    for topic, val in parsed["Topics"].items():
                        if output_schema == "topic_only":
                            topic_rows.append({"Topic": topic, "Present": val})
                        elif output_schema == "sentiment_only":
                            topic_rows.append({"Topic": topic, "Sentiment": val})
                        elif output_schema in ("span_only", "full"):
                            if isinstance(val, list):
                                for item in val:
                                    if item.get("text"):
                                        row_data = {"Topic": topic, "Text": item["text"]}
                                        if "label" in item:
                                            row_data["Label"] = item["label"]
                                        topic_rows.append(row_data)
                    if topic_rows:
                        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)
                    else:
                        st.info("All topics null.")
                else:
                    st.code(r.get("raw_output", "(empty)")[:500])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: Run Experiments
# ═══════════════════════════════════════════════════════════════════════════
with tab_run:
    st.subheader("Run Experiments")
    st.caption("Launch experiments directly from the dashboard. Output will appear in the Inspect tab after refresh.")

    exps = get_experiments()
    exp_names = list(exps.keys())

    col_left, col_right = st.columns(2)
    with col_left:
        selected_exp = st.selectbox("Experiment", exp_names)
        model_override = st.text_input("Model override (blank = use model.yaml)", value="llama4:scout")
        n_reviews = st.number_input("Number of reviews", min_value=1, max_value=500, value=20)
    with col_right:
        if selected_exp:
            cfg = exps[selected_exp]
            st.markdown(f"**Description:** {cfg.get('description', '')}")
            st.markdown(f"**Rules:** {cfg.get('rules', [])}")
            st.markdown(f"**Subtasks:** {cfg.get('subtasks', [])}")
            st.markdown(f"**Schema:** {cfg.get('output_schema', 'full')}")

    run_all = st.checkbox("Run ALL experiments (sequential)")

    if st.button("▶️ Run", type="primary"):
        script = TASK1_DIR / "run_task1.py"
        python = sys.executable
        if run_all:
            experiments_to_run = exp_names
        else:
            experiments_to_run = [selected_exp]

        for exp in experiments_to_run:
            cmd = [python, str(script), "--experiment", exp, "--n", str(n_reviews)]
            if model_override.strip():
                cmd += ["--model", model_override.strip()]
            st.markdown(f"**Running:** `{' '.join(cmd)}`")
            with st.spinner(f"Running {exp}…"):
                result = subprocess.run(
                    cmd,
                    capture_output=True, text=True,
                    cwd=str(TASK1_DIR),
                )
            if result.returncode == 0:
                st.success(f"✅ {exp} — done")
                with st.expander("stdout"):
                    st.text(result.stdout[-3000:] if result.stdout else "(empty)")
            else:
                st.error(f"❌ {exp} — failed (exit {result.returncode})")
                with st.expander("stderr"):
                    st.text(result.stderr[-3000:] if result.stderr else "(empty)")

        st.cache_data.clear()
        st.info("Results refreshed. Switch to Overview or Metrics tab.")
