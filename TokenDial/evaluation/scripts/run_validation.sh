#!/usr/bin/env bash
# Run from the TokenDial repository after installing the documented environment.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$(dirname "$REPO")"
cd "$REPO"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HOME="$BASE/cache/huggingface"
export TORCH_HOME="$BASE/cache/torch"
export DIFFSYNTH_MODEL_BASE_PATH="$BASE/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-$BASE/.venv/bin/python}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
# Set this explicitly to a GPU allocated/available for your account.
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the assigned GPU UUID or index}"
ART="${ARTIFACT_DIR:-$BASE/artifacts/person_older_v2v}"
if [ -d "$ART" ] && [ -n "$(find "$ART" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Artifact directory is not empty: $ART. Set ARTIFACT_DIR to a fresh path."
  exit 1
fi
mkdir -p "$ART"
PROMPT='A man in a dark suit and tie stands on a city street at dusk, turning his head, blurred bokeh lights in the background.'
"$PYTHON" -m unittest discover -s evaluation/tests -v > "$ART/unit_tests.log" 2>&1
"$PYTHON" -m pip freeze > "$ART/environment.lock.txt"
git rev-parse HEAD > "$ART/repository_commit.txt"
"$PYTHON" - <<'PY' > "$ART/hardware.json"
import torch,json,os
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
                  'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))
PY
for SEED in 0 1; do
  "$PYTHON" -m tokendial.v2v \
    --ckpt checkpoints/person_older.safetensors \
    --input_video assets/demo_videos/man_real.mp4 \
    --prompt "$PROMPT" --rgb_cfg_scales '0,1,2,3,4' \
    --seed "$SEED" --num_frames 17 --height 320 --width 576 \
    --T_steps 50 --n_max 40 --n_min 0 --n_avg 1 \
    --output_dir "$ART/videos/seed_$SEED" > "$ART/generation_seed_$SEED.log" 2>&1
done
"$PYTHON" -m evaluation adapt --root "$ART/videos" --manifest "$ART/manifest.json" \
  --seeds '0,1' --model-scales '0,1,2,3,4' --concept person --attribute older \
  --checkpoint checkpoints/person_older.safetensors --source-video assets/demo_videos/man_real.mp4 \
  --prompt "$PROMPT" --notes 'V2V integration validation only; 17 frames, 320x576, 15fps output, 50 steps, n_max40, no mask; not the paper benchmark'
"$PYTHON" -m evaluation validate --manifest "$ART/manifest.json" --guidance-dir evaluation/guidance --probe
"$PYTHON" -m evaluation evaluate --manifest "$ART/manifest.json" --guidance-dir evaluation/guidance --out-dir "$ART/candidate" --device cuda
"$PYTHON" -m evaluation align --manifest "$ART/manifest.json" --guidance-dir evaluation/guidance \
  --metrics-csv "$ART/candidate/metrics.csv" --out-dir "$ART/alignment" --device cuda
echo VALIDATION_COMPLETE
