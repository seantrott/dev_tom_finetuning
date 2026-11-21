#!/usr/bin/env bash
# Run the fine-tuning transfer analysis at each training stage of OLMo-2 13B:
# Stage 1, Stage 2, Pretrained (final base model), SFT, DPO, Instruct.
# Run from the repo root, inside tmux:
#   bash run_revisions.sh
#
# Each run writes to data/processed/finetune/<model>/<revision>/ and skips
# runs that already have output, so it is safe to re-run after a crash.

set -e

# One line per point: "<model> <revision>"
#   last-stage1  last stage-1 checkpoint (same pick as run_fb_control_final_checkpoint.py)
#   last-stage2  highest-step stage-2 checkpoint
#   main         the released model
RUNS=(
  "allenai/OLMo-2-1124-13B last-stage1"
  "allenai/OLMo-2-1124-13B last-stage2"
  "allenai/OLMo-2-1124-13B main"
  "allenai/OLMo-2-1124-13B-SFT main"
  "allenai/OLMo-2-1124-13B-DPO main"
  "allenai/OLMo-2-1124-13B-Instruct main"
)

for RUN in "${RUNS[@]}"; do
  read -r MODEL REV <<< "${RUN}"
  echo "===== ${MODEL} @ ${REV} ====="
  python src/finetune/finetune_transfer.py --model "${MODEL}" --revision "${REV}"
done