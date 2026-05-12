"""Run all Task 1 experiments in sequence with llama3.2 (or any model).

Usage:
  python run_all_experiments.py
  python run_all_experiments.py --model llama4:scout
  python run_all_experiments.py --runs 1         # single run per ID
  python run_all_experiments.py --n 5            # quick test with 5 IDs
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src import load_model_config, load_paths_config, load_topics_config
from src import build_client
from src import run_task1
from src.prompts import build_modular_prompt

EXPERIMENT_ORDER = ["subtask1_1", "subtask1_2", "subtask1_3", "combined"]


def _load_experiments(repo_root: Path) -> dict:
    cfg_path = repo_root / "configs" / "experiments.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("experiments", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Task 1 experiments sequentially")
    parser.add_argument("--model", default=None, help="Override model from model.yaml")
    parser.add_argument("--runs", type=int, default=3, help="Runs per ID (default: 3)")
    parser.add_argument("--n", type=int, default=None, help="Limit IDs for quick testing")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    topics_cfg = load_topics_config(repo_root)
    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    if args.model:
        model_cfg = replace(model_cfg, task1_model=args.model, validator_model=args.model)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)
    experiments = _load_experiments(repo_root)

    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")

    for exp_name in EXPERIMENT_ORDER:
        if exp_name not in experiments:
            print(f"[SKIP] '{exp_name}' not found in experiments.yaml")
            continue

        exp = experiments[exp_name]
        rule_list = [int(r) for r in exp["rules"]]
        output_schema = exp["output_schema"]
        description = exp.get("description", "")
        prompt_template = build_modular_prompt(repo_root, rule_list, output_schema)
        output_subdir = f"{model_folder}/modular/{exp_name}"

        print("\n" + "=" * 60)
        print(f"EXPERIMENT : {exp_name}  ({EXPERIMENT_ORDER.index(exp_name)+1}/{len(EXPERIMENT_ORDER)})")
        print(f"Model      : {model_cfg.task1_model}")
        print(f"Rules      : {rule_list}")
        print(f"Schema     : {output_schema}")
        print(f"Description: {description}")
        print(f"Dataset    : {'all IDs' if args.n is None else f'first {args.n} IDs (test mode)'}")
        print(f"Runs/ID    : {args.runs}")
        print("=" * 60)

        out_path = run_task1(
            repo_root=repo_root,
            client=client,
            topics_cfg=topics_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            limit_reviews=args.n,
            prompt_template=prompt_template,
            output_subdir=output_subdir,
            output_label=exp_name,
            output_schema=output_schema,
            runs_per_id=args.runs,
        )
        print(f"Wrote: {out_path}")

    print("\n" + "=" * 60)
    print("All experiments done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
