#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_experiments_3x.sh
#
# Runs all 17 experiments × 3 runs × 3 data samples with llama4:scout.
#
# Output folder structure per experiment:
#   outputs/task1/llama4_scout/modular/<experiment>/
#     sample_A_run1/   ← reviews  1-20,  run 1
#     sample_A_run2/   ← reviews  1-20,  run 2
#     sample_A_run3/   ← reviews  1-20,  run 3
#     sample_B_run1/   ← reviews 21-40,  run 1
#     sample_B_run2/   ← reviews 21-40,  run 2
#     sample_B_run3/   ← reviews 21-40,  run 3
#     sample_C_run1/   ← reviews 41-60,  run 1
#     sample_C_run2/   ← reviews 41-60,  run 2
#     sample_C_run3/   ← reviews 41-60,  run 3
#
# Usage:
#   ./run_experiments_3x.sh                  # default: n=20, model=llama4:scout
#   ./run_experiments_3x.sh --n 10           # smaller n for quick test
#   ./run_experiments_3x.sh --model llama3.2
#   ./run_experiments_3x.sh --samples AB     # only run samples A and B
#   ./run_experiments_3x.sh --runs 12        # only run runs 1 and 2
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK1_DIR="$SCRIPT_DIR/Task_1"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
if [ -f "$VENV_PYTHON" ]; then
  PYTHON="$VENV_PYTHON"
else
  PYTHON="${PYTHON:-python3}"
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
N=20
MODEL="llama4:scout"
SAMPLES="ABC"   # A=offset 0, B=offset N, C=offset 2N
RUNS="123"

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
  case "$1" in
    --n)       N="$2";       shift; shift ;;
    --model)   MODEL="$2";   shift; shift ;;
    --samples) SAMPLES="$2"; shift; shift ;;
    --runs)    RUNS="$2";    shift; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# ── Experiment list ───────────────────────────────────────────────────────────
ALL_EXPS="combined subtask1_1 subtask1_2 subtask1_3 subtask1_1_and_1_2 subtask1_1_and_1_3 subtask1_2_and_1_3 rules_1_2_5 rules_3_6 rules_4_7 rule1_only rule2_only rule3_only rule4_only rule5_only rule6_only rule7_only"

# Count experiments
TOTAL_EXPS=0
for e in $ALL_EXPS; do TOTAL_EXPS=$((TOTAL_EXPS + 1)); done

TOTAL_RUNS=$(( TOTAL_EXPS * ${#SAMPLES} * ${#RUNS} ))
DONE=0
PASS=0
FAIL=0
FAILED_LIST=""

# Helper: offset for sample letter
sample_offset() {
  local s="$1"
  case "$s" in
    A) echo 0 ;;
    B) echo "$N" ;;
    C) echo "$((N * 2))" ;;
    D) echo "$((N * 3))" ;;
    *) echo 0 ;;
  esac
}

printf "\n════════════════════════════════════════════════════════\n"
printf "  ABA Mining — 3x3 Experiment Run\n"
printf "  Model   : %s\n" "$MODEL"
printf "  Reviews : %s per sample\n" "$N"
printf "  Samples : %s  (A=offset 0, B=offset %s, C=offset %s)\n" "$SAMPLES" "$N" "$((N*2))"
printf "  Runs    : %s\n" "$RUNS"
printf "  Total   : %s runs  (%s experiments)\n" "$TOTAL_RUNS" "$TOTAL_EXPS"
printf "════════════════════════════════════════════════════════\n"

for EXP in $ALL_EXPS; do
  si=0
  while [ $si -lt ${#SAMPLES} ]; do
    SAMPLE="${SAMPLES:$si:1}"
    OFFSET="$(sample_offset "$SAMPLE")"
    si=$((si + 1))

    ri=0
    while [ $ri -lt ${#RUNS} ]; do
      RUN="${RUNS:$ri:1}"
      ri=$((ri + 1))
      DONE=$((DONE + 1))

      printf "\n────────────────────────────────────────────────────────\n"
      printf "  [%3d/%3d]  %-28s  sample_%s  run%s\n" "$DONE" "$TOTAL_RUNS" "$EXP" "$SAMPLE" "$RUN"
      printf "────────────────────────────────────────────────────────\n"

      if "$PYTHON" "$TASK1_DIR/run_task1.py" \
           --experiment "$EXP" \
           --model      "$MODEL" \
           --n          "$N" \
           --offset     "$OFFSET" \
           --sample     "$SAMPLE" \
           --run        "$RUN" 2>&1; then
        printf "  OK  %s / sample_%s_run%s\n" "$EXP" "$SAMPLE" "$RUN"
        PASS=$((PASS + 1))
      else
        printf "  FAIL  %s / sample_%s_run%s\n" "$EXP" "$SAMPLE" "$RUN"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST $EXP/sample_${SAMPLE}_run${RUN}"
      fi
    done
  done
done

printf "\n════════════════════════════════════════════════════════\n"
printf "  SUMMARY\n"
printf "  Passed : %s / %s\n" "$PASS" "$TOTAL_RUNS"
printf "  Failed : %s / %s\n" "$FAIL" "$TOTAL_RUNS"
if [ -n "$FAILED_LIST" ]; then
  printf "  Failed runs:\n"
  for e in $FAILED_LIST; do
    printf "    x %s\n" "$e"
  done
fi
printf "════════════════════════════════════════════════════════\n"
