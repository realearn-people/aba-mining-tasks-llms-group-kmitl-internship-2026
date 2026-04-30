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

INDIVIDUAL_PROMPTS = [
    "prompts/task1/Contrastive/generator_v1.txt",
    "prompts/task1/Contrastive/generator_contrastive1.txt",
    "prompts/task1/Contrastive/generator_contrastive2.txt",
    "prompts/task1/Contrastive/generator_contrastive3.txt",
]

COMBINED_PROMPT = "prompts/task1/Contrastive/generator_combined.txt"


def _load_experiments(repo_root: Path) -> dict:
    cfg_path = repo_root / "configs" / "experiments.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("experiments", {})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 1 mining pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy mode — run existing monolithic prompts
  python3 run_task1.py --mode combined --n 20
  python3 run_task1.py --mode both --n 20

  # Modular mode — assemble prompt from individual rule files
  python3 run_task1.py --experiment combined          # rules 1-7 (all subtasks)
  python3 run_task1.py --experiment subtask1_1        # rules 2,5,6 (topic only)
  python3 run_task1.py --experiment subtask1_2        # rules 1,7   (span only)
  python3 run_task1.py --experiment subtask1_3        # rules 3,4   (sentiment only)
  python3 run_task1.py --rules 1,2,5                  # ad-hoc custom combination

  # Override model for any mode
  python3 run_task1.py --experiment combined --model llama4:scout --n 20
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["individual", "combined", "both"],
        default=None,
        help="Legacy mode: run monolithic generator_*.txt prompts.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Modular mode: name of an experiment from configs/experiments.yaml.",
    )
    parser.add_argument(
        "--rules",
        default=None,
        help="Modular mode (ad-hoc): comma-separated rule numbers, e.g. '1,2,5'.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of reviews to process (default: 20)")
    parser.add_argument(
        "--model",
        default=None,
        help="Override model from model.yaml, e.g. --model llama4:scout",
    )
    args = parser.parse_args()

    # Validate argument combinations
    modular = args.experiment or args.rules
    if not modular and args.mode is None:
        args.mode = "both"  # backwards-compatible default
    if modular and args.mode:
        parser.error("Use either --mode OR (--experiment / --rules), not both.")
    if args.experiment and args.rules:
        parser.error("Use either --experiment OR --rules, not both.")

    repo_root = Path(__file__).resolve().parent
    load_dotenv(repo_root / ".env", override=False)

    topics_cfg = load_topics_config(repo_root)
    model_cfg = load_model_config(repo_root)
    paths_cfg = load_paths_config(repo_root)

    if args.model:
        model_cfg = replace(model_cfg, task1_model=args.model, validator_model=args.model)

    client = build_client(model_cfg.provider, ollama_options=model_cfg.ollama_options)

    model_folder = model_cfg.task1_model.replace(":", "_").replace("/", "_").replace("-", "_")

    # ── MODULAR MODE ─────────────────────────────────────────────────────────
    if modular:
        if args.experiment:
            experiments = _load_experiments(repo_root)
            if args.experiment not in experiments:
                available = ", ".join(experiments.keys())
                parser.error(f"Unknown experiment '{args.experiment}'. Available: {available}")
            exp = experiments[args.experiment]
            rule_list = [int(r) for r in exp["rules"]]
            output_schema = exp.get("output_schema", "full")
            exp_label = args.experiment
            description = exp.get("description", "")
        else:
            rule_list = [int(r.strip()) for r in args.rules.split(",")]
            output_schema = "full"  # Default for ad-hoc rules
            exp_label = f"rules_{'_'.join(str(r) for r in rule_list)}"
            description = f"Ad-hoc rules: {rule_list}"

        print("\n" + "=" * 60)
        print(f"MODE: modular  |  model: {model_cfg.task1_model}")
        print(f"Experiment : {exp_label}")
        print(f"Rules      : {rule_list}")
        print(f"Output schema: {output_schema}")
        print(f"Description: {description}")
        print("=" * 60)

        prompt_template = build_modular_prompt(repo_root, rule_list, output_schema)

        out_path = run_task1(
            repo_root=repo_root,
            client=client,
            topics_cfg=topics_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            limit_reviews=args.n,
            prompt_template=prompt_template,
            output_subdir=f"{model_folder}/modular/{exp_label}",
            output_label=exp_label,
            output_schema=output_schema,
        )
        print(f"Wrote: {out_path}")

    # ── LEGACY MODE ───────────────────────────────────────────────────────────
    else:
        if args.mode in ("individual", "both"):
            print("\n" + "=" * 60)
            print(f"MODE: individual  |  model: {model_cfg.task1_model}")
            print("=" * 60)
            for prompt_path in INDIVIDUAL_PROMPTS:
                prompt_label = Path(prompt_path).stem
                print(f"\n--- Running with prompt: {prompt_label} ---")
                out_path = run_task1(
                    repo_root=repo_root,
                    client=client,
                    topics_cfg=topics_cfg,
                    model_cfg=model_cfg,
                    paths_cfg=paths_cfg,
                    limit_reviews=args.n,
                    prompt_path=prompt_path,
                    output_subdir=f"{model_folder}/individual",
                )
                print(f"Wrote: {out_path}")

        if args.mode in ("combined", "both"):
            print("\n" + "=" * 60)
            print(f"MODE: combined  |  model: {model_cfg.task1_model}")
            print("=" * 60)
            prompt_label = Path(COMBINED_PROMPT).stem
            print(f"\n--- Running with prompt: {prompt_label} ---")
            out_path = run_task1(
                repo_root=repo_root,
                client=client,
                topics_cfg=topics_cfg,
                model_cfg=model_cfg,
                paths_cfg=paths_cfg,
                limit_reviews=args.n,
                prompt_path=COMBINED_PROMPT,
                output_subdir=f"{model_folder}/combined",
            )
            print(f"Wrote: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
