#!/usr/bin/env bash
# Run the fine-tuning transfer analysis at each training stage:
# Stage 1, Stage 2, Pretrained (final base model), SFT, DPO, Instruct.
# Run from the repo root, inside tmux:
#   bash run.sh
#
# Each run writes to data/processed/finetune/<model>/<revision>/ and skips
# runs that already have output, so it is safe to re-run after a crash.
#
# The finished 13B checkpoints are left out on purpose: the script loads the
# model before checking which runs are done, so listing them again would
# re-download them just to skip them.

set -e

# One line per point: "<model> <revision>"
#   last-stage1  last stage-1 checkpoint
#   last-stage2  highest-step stage-2 checkpoint
#   main         the released model
RUNS=(
  "allenai/OLMo-2-1124-13B-Instruct main"
  "allenai/OLMo-2-1124-7B last-stage1"
  "allenai/OLMo-2-1124-7B last-stage2"
  "allenai/OLMo-2-1124-7B main"
  "allenai/OLMo-2-1124-7B-SFT main"
  "allenai/OLMo-2-1124-7B-DPO main"
  "allenai/OLMo-2-1124-7B-Instruct main"
)

for RUN in "${RUNS[@]}"; do
  read -r MODEL REV <<< "${RUN}"
  echo "===== ${MODEL} @ ${REV} ====="
  python src/finetune/finetune_transfer.py --model "${MODEL}" --revision "${REV}"
  # Free disk space once this checkpoint is done (results in data/processed are kept)
  rm -rf ~/.cache/huggingface/hub
done