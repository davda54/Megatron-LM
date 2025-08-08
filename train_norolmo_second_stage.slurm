#!/bin/bash
#SBATCH --job-name=LLM
#SBATCH --account=nn10029k
#SBATCH --partition=accel
#SBATCH --time=48:00:00
#SBATCH --nodes=32
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

module purge
module load PrgEnv-nvidia/8.6.0
module load craype-arm-grace
module load gcc-native/13.2
module load cuda/12.6
module load cray-python/3.11.7
module load nvhpc/24.11


source /cluster/projects/nn9851k/pytorch_2.7.0/bin/activate
export HF_HOME="/cluster/projects/nn9851k/hf_cache/"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/

export CUDNN_PATH=/cluster/projects/nn9851k/pytorch_2.7.0/lib/python3.11/site-packages/nvidia/cudnn
export CPLUS_INCLUDE_PATH=/cluster/projects/nn9851k/pytorch_2.7.0/lib/python3.11/site-packages/nvidia/cudnn/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=/cluster/projects/nn9851k/pytorch_2.7.0/lib/python3.11/site-packages/nvidia/cudnn/lib/:$LD_LIBRARY_PATH
export CXX=/opt/cray/pe/gcc-native/13/bin/g++ 
export CC=/opt/cray/pe/gcc-native/13/bin/gcc 
export LD=/opt/cray/pe/gcc-native/13/bin/g++

# This is a slurm script for training generative models on LUMI using
# Megatron-LM pretrain_gpt.py. This script defines defaults for
# training a FineWeb-like model (approx. 1.7B parameters) and is
# intended to be reasonably easily modified for other model sizes
# by editing the variables defined in the "MODEL AND PRETRAINING
# CONFIGURATION" section below.
#
# Note that while the script defines default arguments for sbatch
# in the #SBATCH comments above, you can override any of these on the
# command line. For example, to run on 16 nodes:
#
#    sbatch --nodes 16 ./train.sh [...]

######################################################################
#
# ENVIRONMENT SETUP AND GENERAL CONFIGURATION
#
# This section of the script sets up the execution environment (logs,
# container, etc.) and configuration that is independent of the model
# or pretraining setup. It should generally not be necessary to edit
# this section, and you may wish to double-check that you understand
# what you are doing before you do.
#
######################################################################

# If this script is run without sbatch, invoke with sbatch here. This
# also gives us an opportunity to make sure logs/ exists. (If the
# directory where --output and/or --error are directed to doesn't
# exist, the run will fail silently.)
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0" "$@"
    exit
fi

# Bash "strict mode"
# (see http://redsymbol.net/articles/unofficial-bash-strict-mode/)
set -euo pipefail

# When slurm reschedules a job that ended on node failure, it will run
# with the same job ID, clobbering the original logs. Rename the logs
# and include timestamp to avoid this.
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logfile_basename="${SLURM_JOB_NAME}-${SLURM_JOBID}-${timestamp}"

# PATHS
BASE_DIR="$SLURM_SUBMIT_DIR"
OUTPUT_DIR="$BASE_DIR/output_second_stage"
CHECKPOINT_PATH="$OUTPUT_DIR/checkpoints_second_stage"
TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard/$SLURM_JOB_NAME-$SLURM_JOBID"

mkdir -p "$CHECKPOINT_PATH"    # This needs to exist

# Script that is used to launch on GPU nodes
LAUNCH_SCRIPT="$BASE_DIR/launch.sh"

# Needed for sequence paralellism
# (see https://github.com/NVIDIA/Megatron-LM/issues/533)
export CUDA_DEVICE_MAX_CONNECTIONS=1

# DISTRIBUTED ARGS
# These are used by torch.distributed to allow the different processes
# to find each other. Note that RANK and LOCAL_RANK are also expected,
# but can only be set in the launcher script as the values are
# specific to the process.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=8888    # TODO add in job ID
export WORLD_SIZE=$SLURM_NTASKS    # Note: only valid if ntasks==ngpus

# NVIDIA GH200 Optimizations
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1

# NCCL Configuration for NVIDIA
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_LEVEL=NVL
export NCCL_TREE_THRESHOLD=0
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=0
# Update with your actual network interfaces:
# export NCCL_SOCKET_IFNAME=eth0,ib0  

# Threading optimization for Grace CPU
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Distributed settings
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((8888 + SLURM_JOBID % 1000))
export WORLD_SIZE=$SLURM_NTASKS

######################################################################
#
# MODEL AND PRETRAINING CONFIGURATION
#
# This section sets variables that define the model and pretraining
# configuration. These mostly correspond to command-line arguments to
# Megatron-LM/pretrain_gpt.py, and when they do, the names should
# match (e.g. the variable $GLOBAL_BATCH_SIZE gets passed as
# --global-batch-size). This script is intended to be configurable by
# redefining these variables.
#
######################################################################

# DATA
DATA_ROOT="${BASE_DIR}/../Megatron-LM/tokenized_corpora"
DATA_PATH="1.0 ${DATA_ROOT}/norolmo_hplt_nob_13_text_document"
DATA_CACHE_PATH="$DATA_ROOT/cache_david"
TOKENIZER_MODEL="/cluster/projects/nn9851k/Megatron-LM/tokenizers/norwegian_olmo"

# MODEL
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=32    # No GQA when NUM_QUERY_GROUPS=NUM_ATTENTION_HEADS
TIE_WORD_EMBEDDINGS=0
INIT_METHOD_STD=0.02
SEQ_LENGTH=4096
ROTARY_BASE=500000    # Default, recommend larger for higher seq len

# PARALLELISM
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE=1
PROFILE=0

# OPTIMIZER
ADAM_BETA1=0.9
ADAM_BETA2=0.95
ADAM_EPS=1e-8
LR=1e-4
MIN_LR=1e-4
LR_WARMUP_ITERS=10
CLIP_GRAD=100.0
WEIGHT_DECAY=1e-1

# TRAINING
FSDP=0
GLOBAL_BATCH_SIZE=1024
MICRO_BATCH_SIZE=2
RECOMPUTATION=0
TRAIN_TOKENS=100663296000    # TRAIN_SAMPLES computed from this

# SAVING AND EVALUATION
LOG_INTERVAL=10
SAVE_INTERVAL=100000
EVAL_INTERVAL=500000
EVAL_ITERS=1

######################################################################
#
# DERIVED CONFIGURATION SETTINGS
#
# The following settings are derived from the configuration above.
# Do set these directly, as they will be overwritten here.
#
######################################################################

# Check that variables are not set (sanity)
confirm_unset() {
    local varname="$1"
    if [ -n "${!varname+x}" ]; then
	echo "Error: variable '$varname' should not be set." >&2
	exit 1
    fi
}
confirm_unset "TRAIN_SAMPLES"
confirm_unset "LR_WARMUP_SAMPLES"
confirm_unset "LR_DECAY_SAMPLES"

# Calculate TRAIN_SAMPLES from TRAIN_TOKENS
TRAIN_TOKENS=${TRAIN_TOKENS//_}    # drop "_" for bash math
TRAIN_SAMPLES=$((TRAIN_TOKENS/SEQ_LENGTH))

# Set LR_WARMUP_SAMPLES and LR_DECAY_SAMPLES and based LR_WARMUP_ITERS
# and TRAIN_SAMPLES
LR_WARMUP_SAMPLES=$((LR_WARMUP_ITERS*GLOBAL_BATCH_SIZE))
LR_DECAY_SAMPLES=$TRAIN_SAMPLES

######################################################################
#
# BUILDING COMMAND-LINE ARGUMENTS
#
# The following builds the command-line arguments for
# Megatron-LM/pretrain_gpt.py based on the variables defined above
# (and optionally in any config given to the script). Note that some
# arguments that are not expected to vary are hard-coded here.
#
######################################################################

DATA_ARGS=(
    --data-path "$DATA_PATH"
    --data-cache-path "$DATA_CACHE_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_MODEL"
    --make-vocab-size-divisible-by 128
    --dataloader-type single
    --num-workers 2   # Some issues with this, lower values are safer
)

MODEL_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
)

if [ "$NUM_QUERY_GROUPS" != "$NUM_ATTENTION_HEADS" ]; then
    MODEL_ARGS+=(
        --group-query-attention
        --num-query-groups $NUM_QUERY_GROUPS
    )
fi

if [ "$TIE_WORD_EMBEDDINGS" = "0" ]; then
    MODEL_ARGS+=(
	--untie-embeddings-and-output-weights
    )
fi

if [ "$FSDP" = "1" ]; then
    PARALLEL_ARGS=(
	--use-torch-fsdp2
	--ckpt-format torch_dist
    )
else
    PARALLEL_ARGS=(
	--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE
	--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE
	--context-parallel-size $CONTEXT_PARALLEL_SIZE
	--use-distributed-optimizer
    )
fi

if [ "$PROFILE" = "1" ]; then
    PROFILE_ARGS=(
	--use-pytorch-profiler
	--profile-ranks 0
	--profile-step-start 5
	--profile-step-end 7
    )
else
    PROFILE_ARGS=()
fi

MODEL_ARGS+=(
    --use-flash-attn
    --attention-softmax-in-fp32
    --max-position-embeddings $SEQ_LENGTH
    --seq-length $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base $ROTARY_BASE
    --disable-bias-linear
    --init-method-std $INIT_METHOD_STD
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples $TRAIN_SAMPLES
    --bf16
    --swiglu
    --distributed-timeout-minutes 30
    --overlap-grad-reduce
    --norm-epsilon 0.000001
    --post-layer-norm
    --qk-layernorm
    --freeze_transformer
    --freeze_iters 1024000
    --hf_checkpoint OLMo-2-1224-7B_remapped
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --adam-beta1 $ADAM_BETA1
    --adam-beta2 $ADAM_BETA2
    --adam-eps $ADAM_EPS
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-style WSD
    --lr-wsd-decay-style linear
    --lr-decay-samples 10000000000000
    --lr-wsd-decay-samples 4915200
    --lr-warmup-samples $LR_WARMUP_SAMPLES
    --clip-grad $CLIP_GRAD
    --weight-decay $WEIGHT_DECAY
)

OUTPUT_ARGS=(
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --tensorboard-dir "$TENSORBOARD_DIR"
    --tensorboard-queue-size 5
    --log-throughput
    --log-progress
    --wandb-project "normistral"
    --wandb-exp-name "norolmo"
    --wandb-save-dir "./wandb"
    --log-interval $LOG_INTERVAL
)

# Interleaved pipeline scheduling is only possible with pipeline
# parallel degree > 1.
if [ $PIPELINE_MODEL_PARALLEL_SIZE -gt 1 ] && [ $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE -gt 1 ]; then
    PARALLEL_ARGS+=(
	--num-layers-per-virtual-pipeline-stage $NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE
    )
fi

if [ "$RECOMPUTATION" = "1" ]; then
    MODEL_ARGS+=(
	--recompute-activations
	--recompute-granularity selective
    )
fi

CHECKPOINT_ARGS=(
    --ckpt-format torch    # "legacy" checkpoints; torch_dist is crashing
#    --async-save    # requires --ckpt-format torch_dist
    --load "$CHECKPOINT_PATH"
    --save "$CHECKPOINT_PATH"
    --save-interval $SAVE_INTERVAL
)

COMMAND=" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
"

######################################################################
#
# Run the command through the launch script with srun.
# Note that any node-specific setup needs to go into the launch script.
#
######################################################################

echo '============= COMMAND: ============='
echo "$COMMAND"
echo '===================================='

echo "START $SLURM_JOBID: $(date)"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

srun \
    --cpu-bind=verbose,cores \
    --distribution=block:block \
    --label \
    "$LAUNCH_SCRIPT" \
    $COMMAND
#    --cpu-bind=mask_cpu:$BIND_MASK \
echo "END $SLURM_JOBID: $(date)"
