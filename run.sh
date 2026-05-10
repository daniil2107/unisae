#!/usr/bin/env bash
#
# Universal SAE across 3 small models from different families.
# Models:
#   gpt2                          (124M, 12 layers, GPT-2 BPE)
#   EleutherAI/pythia-160m        (160M, 12 layers, GPT-NeoX BPE)
#   HuggingFaceTB/SmolLM-135M     (135M, 30 layers, SmolLM BPE — different family)
#
# The aligned precompute tokenizes each text chunk with each model's own
# tokenizer, runs each model's forward, and aligns activations via
# src.tokenizer_align.align_n_models (Algorithm 1 of arXiv:2602.11729
# extended to N streams). Output shards carry `alignment_mode="greedy"` in
# store_config.json so dataset.py reads the aligned format.

set -euo pipefail

STORE_DIR="${STORE_DIR:-/ephemeral/spotai_small/dognev/unisae_store}"
NUM_TOKENS="${NUM_TOKENS:-10000000}"       # 10M aligned tokens for a first run; bump to 100M for real training

# 1) Precompute aligned activations.
python -m src.precompute_aligned \
  --model-names gpt2 EleutherAI/pythia-160m HuggingFaceTB/SmolLM-135M \
  --hook-points blocks.6.hook_resid_post blocks.6.hook_resid_post blocks.14.hook_resid_post \
  --num-aligned-tokens "${NUM_TOKENS}" \
  --primary-sequence-length 512 \
  --batch-size 8 \
  --shard-size-tokens 262144 \
  --output-dir "${STORE_DIR}" \
  --dtype float16 \
  --device cuda \
  --prepend-bos

# 2) Train the universal SAE on the aligned activations.
python -m src.sparc_train \
  --store-dir "${STORE_DIR}" \
  --latent-dim 16384 \
  --lr 3e-4 \
  --num-epochs 5 \
  --batch-size 4096 \
  --num-workers 4 \
  --recon-weight 1.0 \
  --cross-recon-weight 1.0 \
  --k 128 \
  --calib-batches 256 \
  --device cuda \
  --save-path "${STORE_DIR}/unisae_ckpt.pt"
