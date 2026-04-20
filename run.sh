# Pre-compute activations for models and save them to disk.

python  -m src.precompute_activations \
  --model-names pythia-410m pythia-1b \
  --hook-point blocks.10.hook_resid_post \
  --num-tokens 10485760 \
  --sequence-length 256 \
  --batch-size 8 \
  --shard-size-tokens 262144 \
  --output-dir /Vrac/renton/pythia_activation \
  --dtype float16 \
  --num-workers 4 \
  --device cuda

# Train from the pre-computed activations.

python -m src.sparc_train \ 
    --store-dir /Vrac/renton/pythia_activation \
    --latent-dim 8192 \
    --lr 3e-4 \
    --num-epochs 1 \
    --batch-size 256 \
    --num-workers 4 \
    --recon-weight 1.0 \
    --cross-recon-weight 1.0 \
    --k 128 \
    --device cuda