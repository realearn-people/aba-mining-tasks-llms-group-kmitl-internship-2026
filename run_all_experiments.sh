#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all_experiments.sh
# Run every experiment defined in experiments.yaml with llama4:scout.
#
# Usage:
#   ./run_all_experiments.sh              # default: 20 reviews each
#   ./run_all_experiments.sh --n 5        # quick smoke-test with 5 reviews
#   ./run_all_experiments.sh --n 20 --model llama3.2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK1_DIR="$SCRIPT_DIR/Task_1"
PYTHON="${PYTHON:-python3}"

# ── Defaults ──────────────────────────────────────────────────────────────────
N=20
MODEL="llama4:scout"

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)    N="$2";     shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

echo "========================================"
echo "  ABA Mining — Full Experiment Run"
echo "  Model  : $MODEL"
echo "  Reviews: $N"
echo "========================================"
echo ""

# Ordered list of experiments
# ── Category 1: Baseline (all rules together) ────────────────────────────────
COMBINED_EXPS=(
  "combined"
)

# ── Category 2: Single-subtask ───────────────────────────────────────────────
SINGLE_SUBTASK_EXPS=(
  "subtask1_1"
  "subtask1_2"
  "subtask1_3"
)

# ── Category 3: Two-subtask combinations ─────────────────────────────────────
TWO_SUBTASK_EXPS=(
  "subtask1_1_and_1_2"
  "subtask1_1_and_1_3"
  "subtask1_2_and_1_3"
)

# ── Category 4: Partial-rule ablations ───────────────────────────────────────
PARTIAL_EXPS=(
  "rules_1_2_5"
  "rules_3_6"
  "rules_4_7"
)

# ── Category 5: Single-rule ablations ────────────────────────────────────────
SINGLE_RULE_EXPS=(
  "rule1_only"
  "rule2_only"
  "rule3_only"
  "rule4_only"
  "rule5_only"
  "rule6_only"
  "rule7_only"
)

ALL_EXPS=(
  "${COMBINED_EXPS[@]}"
  "${SINGLE_SUBTASK_EXPS[@]}"
  "${TWO_SUBTASK_EXPS[@]}"
  "${PARTIAL_EXPS[@]}"
  "${SINGLE_RULE_EXPS[@]}"
)

TOTAL=${#ALL_EXPS[@]}
PASS=0
FAIL=0
FAILED_EXPS=()

run_experiment() {
  local exp="$1"
  local idx="$2"
  echo ""
  echo "────────────────────────────────────────"
  echo "[$idx/$TOTAL] $exp"
  echo "────────────────────────────────────────"
  if $PYTHON "$TASK1_DIR/run_task1.py" \
       --experiment "$exp" \
       --model "$MODEL" \
       --n "$N" 2>&1; then
    echo "✅  $exp — done"
    PASS=$((PASS + 1))
  else
    echo "❌  $exp — FAILED (exit $?)"
    FAIL=$((FAIL + 1))
    FAILED_EXPS+=("$exp")
  fi
}

IDX=0
for EXP in "${ALL_EXPS[@]}"; do
  IDX=$((IDX + 1))
  run_experiment "$EXP" "$IDX"
done

echo ""
echo "========================================"
echo "  SUMMARY"
echo "  Passed : $PASS / $TOTAL"
echo "  Failed : $FAIL / $TOTAL"
if [ ${#FAILED_EXPS[@]} -gt 0 ]; then
  echo "  Failed experiments:"
  for e in "${FAILED_EXPS[@]}"; do
    echo "    - $e"
  done
fi
echo "========================================"
echo ""
echo "To view results, run:"
echo "  cd Dashboard && streamlit run app.py"
