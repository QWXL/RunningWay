#!/bin/bash
################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
################################################################################

export PYTHONUNBUFFERED=1
source /home/chumenta/miniconda3/etc/profile.d/conda.sh

conda activate runningway-train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load configuration from JSON file
CONFIG_FILE="config.json"
echo "[Preload] Load config from $CONFIG_FILE"

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please install jq to parse JSON configuration."
    echo "You can install it with 'sudo apt-get install jq' on Ubuntu/Debian"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Configuration file $CONFIG_FILE not found!"
    exit 1
fi

# Extract model variables from JSON config
MODEL_NAME=$(jq -r '.model.name' "$CONFIG_FILE")

MODEL_TYPE=$(jq -r '.model.type' "$CONFIG_FILE")
N_LAYER=$(jq -r '.model.n_layer' "$CONFIG_FILE")
N_EMBD=$(jq -r '.model.n_embd' "$CONFIG_FILE")
CTX_LEN=$(jq -r '.model.ctx_len' "$CONFIG_FILE")
HEAD_SIZE=$(jq -r '.model.head_size' "$CONFIG_FILE")
VOCAB_SIZE=$(jq -r '.model.vocab_size' "$CONFIG_FILE")
PROJ_DIR=$(jq -r '.paths.proj_dir' "$CONFIG_FILE")

# Extract training variables from JSON config
M_BSZ=$(jq -r '.training.micro_bsz' "$CONFIG_FILE")
LR_INIT=$(jq -r '.training.lr_init' "$CONFIG_FILE")
LR_FINAL=$(jq -r '.training.lr_final' "$CONFIG_FILE")
GRAD_CP=$(jq -r '.training.grad_cp' "$CONFIG_FILE")
GRAD_CLIP=$(jq -r '.training.grad_clip' "$CONFIG_FILE")
EPOCH_SAVE=$(jq -r '.training.epoch_save' "$CONFIG_FILE")
EPOCH_BEGIN=$(jq -r '.training.epoch_begin' "$CONFIG_FILE")
EPOCH_COUNT=$(jq -r '.training.epoch_count' "$CONFIG_FILE")
TRAIN_STAGE=$(jq -r '.training.train_stage' "$CONFIG_FILE")
WARMUP_STEPS=$(jq -r '.training.warmup_steps' "$CONFIG_FILE")
WEIGHT_DECAY=$(jq -r '.training.weight_decay' "$CONFIG_FILE")
ADAM_EPS=$(jq -r '.training.adam_eps' "$CONFIG_FILE")
BETA1=$(jq -r '.training.beta1' "$CONFIG_FILE")
BETA2=$(jq -r '.training.beta2' "$CONFIG_FILE")
LOAD_MODEL=$(jq -r '.training.load_model' "$CONFIG_FILE")

MULTI_STATE=$(jq -r '.training.runningway.use_multi_state' "$CONFIG_FILE")
WINDOW_SIZE=$(jq -r '.training.runningway.window_size' "$CONFIG_FILE")
RESET_STATE_PER_BATCH=$(jq -r '.training.runningway.reset_state_per_batch' "$CONFIG_FILE")
SYSTEM_PROMPT_FROZEN=$(jq -r '.training.runningway.system_prompt_frozen' "$CONFIG_FILE")
STATE_POOL_LEARNING_RATE=$(jq -r '.training.runningway.state_pool_learning_rate' "$CONFIG_FILE")
SYSTEM_RATIO=$(jq -r '.training.runningway.system_ratio' "$CONFIG_FILE")
RNN_RATIO=$(jq -r '.training.runningway.rnn_ratio' "$CONFIG_FILE")
WINDOW_RATIO=$(jq -r '.training.runningway.window_ratio' "$CONFIG_FILE")
USE_NEW_CUDA_KERNEL=$(jq -r '.training.runningway.use_new_cuda_kernel' "$CONFIG_FILE")
FALLBACK_TO_PYTHON=$(jq -r '.training.runningway.fallback_to_python' "$CONFIG_FILE")

# Extract system variables from JSON config
N_NODE=$(jq -r '.system.n_node' "$CONFIG_FILE")
GPU_PER_NODE=$(jq -r '.system.gpu_per_node' "$CONFIG_FILE")
DS_BUCKET_MB=$(jq -r '.system.ds_bucket_mb' "$CONFIG_FILE")
PRECISION=$(jq -r '.system.precision' "$CONFIG_FILE")
STRATEGY=$(jq -r '.system.strategy' "$CONFIG_FILE")
ACCELERATOR=$(jq -r '.system.accelerator' "$CONFIG_FILE")

# Extract data variables from JSON config
MY_EXIT_TOKENS=$(jq -r '.data.my_exit_tokens' "$CONFIG_FILE")
MAGIC_PRIME=$(jq -r '.data.magic_prime' "$CONFIG_FILE")
DATA_FILE=$(jq -r '.data.data_file' "$CONFIG_FILE")
DATA_TYPE=$(jq -r '.data.data_type' "$CONFIG_FILE")

echo "Loaded configuration from $CONFIG_FILE"
echo "Test name: $MODEL_NAME"
echo "Model type: $MODEL_TYPE"
echo "Layers: $N_LAYER"
echo "Embedding dimension: $N_EMBD"

#
################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
# takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
# 1 => slower, save VRAM; 0 => faster, more VRAM
# save every 10 "miniepochs" (1 miniepoch = 40320 * ctx_len tokens) => decrease if your GPU is weak
#
################################################################################
#
# magic_prime = the largest 3n+2 prime smaller than datalen/ctxlen-1 (= 1498226207/512-1 = 2926222.06 in this case) = 2926181 in this case
# use https://www.dcode.fr/prime-numbers-search
#
#
# set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)

python train.py \
 --accelerator $ACCELERATOR \
 --adam_eps $ADAM_EPS \
 --beta1 $BETA1 \
 --beta2 $BETA2 \
 --ctx_len $CTX_LEN \
 --data_file $DATA_FILE \
 --data_type $DATA_TYPE \
 --devices $GPU_PER_NODE \
 --ds_bucket_mb $DS_BUCKET_MB \
 --epoch_begin $EPOCH_BEGIN \
 --epoch_count $EPOCH_COUNT \
 --epoch_save $EPOCH_SAVE \
 --grad_cp $GRAD_CP \
 --grad_clip $GRAD_CLIP \
 --head_size $HEAD_SIZE \
 --load_model $LOAD_MODEL \
 --lr_final $LR_FINAL \
 --lr_init $LR_INIT \
 --magic_prime $MAGIC_PRIME \
 --micro_bsz $M_BSZ \
 --my_exit_tokens $MY_EXIT_TOKENS \
 --my_testing $MODEL_TYPE \
 --n_embd $N_EMBD \
 --n_layer $N_LAYER \
 --num_nodes $N_NODE \
 --precision $PRECISION \
 --proj_dir $PROJ_DIR \
 --strategy $STRATEGY \
 --train_stage 1 \
 --vocab_size $VOCAB_SIZE \
 --warmup_steps $WARMUP_STEPS \
 --weight_decay $WEIGHT_DECAY \
 --use_multi_state $MULTI_STATE \
 --window_size $WINDOW_SIZE \
 --reset_state_per_batch $RESET_STATE_PER_BATCH \
 --system_prompt_frozen $SYSTEM_PROMPT_FROZEN \
 --state_pool_learning_rate $STATE_POOL_LEARNING_RATE \
 --system_ratio $SYSTEM_RATIO \
 --rnn_ratio $RNN_RATIO \
 --window_ratio $WINDOW_RATIO \
 --use_new_cuda_kernel $USE_NEW_CUDA_KERNEL \
 --fallback_to_python $FALLBACK_TO_PYTHON \
 --config $CONFIG_FILE