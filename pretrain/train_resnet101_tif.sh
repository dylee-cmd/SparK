#!/usr/bin/env bash
# SparK pretraining of ResNet-101(-vd) on the [0, 1] float tif patches made by `make_tif_patches.py`, with DDP (torchrun).
#
# Single node, all visible GPUs:
#   bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir
# Single node, GPU 0-3 only:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir
# Multiple nodes (run on every node, NODE_RANK=0 on the master node):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir
# Any hyperparameter below can be overridden by an environment variable, e.g.:
#   MODEL=resnet101 BATCH_PER_GPU=32 EP=800 bash train_resnet101_tif.sh /path/to/patches /path/to/exp_dir
# Extra arguments after the 2nd one are passed to main.py, e.g. `... /path/to/exp_dir --mask=0.75`.
set -euo pipefail

DATA_PATH=${1:?"usage: bash $0 <patch_dir made by make_tif_patches.py> <exp_dir> [extra main.py args]"}
EXP_DIR=${2:?"usage: bash $0 <patch_dir made by make_tif_patches.py> <exp_dir> [extra main.py args]"}
shift 2
DATA_PATH=$(realpath -m "${DATA_PATH}")   # absolute paths, since we `cd` to SparK/pretrain below
EXP_DIR=$(realpath -m "${EXP_DIR}")

# ---------------- model / data ----------------
MODEL=${MODEL:-resnet101vd}           # resnet101vd: Baidu ResNet-101-vd (models/resnetd.py); resnet101: timm ResNet-101
INPUT_SIZE=${INPUT_SIZE:-224}         # must be a multiple of 32 and <= --patch_size of make_tif_patches.py
EXP_NAME=${EXP_NAME:-$(basename "${EXP_DIR}")}

# ---------------- DDP ----------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS_DEFAULT=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NUM_GPUS_DEFAULT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-${NUM_GPUS_DEFAULT}}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "[error] no GPU found (set NPROC_PER_NODE or CUDA_VISIBLE_DEVICES)" >&2; exit 1
fi

# ---------------- hyperparameters (SparK defaults except the batch size) ----------------
BATCH_PER_GPU=${BATCH_PER_GPU:-64}                          # lower it if CUDA out of memory
BS=${BS:-$((BATCH_PER_GPU * NPROC_PER_NODE * NNODES))}      # total batch size of all GPUs
BASE_LR=${BASE_LR:-2e-4}            # actual lr = base_lr * bs / 256 (computed in utils/arg_util.py)
EP=${EP:-1600}
WP_EP=${WP_EP:-40}
MASK=${MASK:-0.6}
DP=${DP:-0.0}                       # 0.0: use the default drop path rate of the model (0.08 for ResNet-101), see models/__init__.py
WORKERS=${WORKERS:-8}               # dataloader workers per GPU

# ---------------- checks ----------------
cd "$(dirname "$0")"     # SparK/pretrain
if [[ ! -d "${DATA_PATH}" ]]; then
    echo "[error] data path not found: ${DATA_PATH}" >&2; exit 1
fi
NUM_PATCHES=$(find -L "${DATA_PATH}" -type f \( -iname '*.tif' -o -iname '*.tiff' \) | head -n $((BS + 1)) | wc -l || true)   # `|| true`: `head` may close the pipe early (SIGPIPE + pipefail)
if [[ "${NUM_PATCHES}" -le "${BS}" ]]; then
    echo "[warning] only ${NUM_PATCHES} patches found in ${DATA_PATH}, which is <= the total batch size ${BS}" >&2
fi

# resume automatically if a checkpoint exists in the exp_dir (main.py sanitizes the exp_dir name the same way)
EXP_DIR_ABS=$(python3 -c "import os,sys; e=os.path.abspath(sys.argv[1]); d,b=os.path.split(e); print(os.path.join(d, ''.join(c if (c.isalnum() or c=='-') else '_' for c in b)))" "${EXP_DIR}")
CKPT="${EXP_DIR_ABS}/${MODEL}_withdecoder_1kpretrained_spark_style.pth"
RESUME_ARGS=()
if [[ -f "${CKPT}" ]]; then
    echo "[info] resume from ${CKPT}"
    RESUME_ARGS=(--resume_from="${CKPT}")
fi

echo "[info] model=${MODEL}, nodes=${NNODES}, gpus/node=${NPROC_PER_NODE}, total bs=${BS}, base_lr=${BASE_LR}, ep=${EP}, exp_dir=${EXP_DIR_ABS}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    --data_path="${DATA_PATH}" \
    --data_type=tif \
    --exp_name="${EXP_NAME}" \
    --exp_dir="${EXP_DIR}" \
    --model="${MODEL}" \
    --input_size="${INPUT_SIZE}" \
    --bs="${BS}" \
    --base_lr="${BASE_LR}" \
    --ep="${EP}" \
    --wp_ep="${WP_EP}" \
    --mask="${MASK}" \
    --dp="${DP}" \
    --dataloader_workers="${WORKERS}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    "$@"

# outputs in ${EXP_DIR}:
#   ${MODEL}_1kpretrained_timm_style.pth            : backbone weights for downstream (for resnet101vd, loadable by mmdet ResNetV1dPaddle depth=101)
#   ${MODEL}_withdecoder_1kpretrained_spark_style.pth : full checkpoint (used to resume)
