##root/run/run_any.sh
#The main script to call when running something on (or off) the cluster
# The general usage is:
# ENV_VAR1=VALUE1 ENV_VAR2=VALUE2 ... bash run_any.sh COMMAND
#
# run_any runs any command (e.g. python, bash, etc) on a remote node with slurm or the local machine depending on the 
# properties defined as env variables
#
# Example usage:
# 1) Run on the current machine withour slurm:
# SLURM=0 ENV_NAME="ivg" bash ./run/run_any.sh python {your_script.py} {--your script --parameters}
#
# 2) Run on a remote node with slurm:
# NODE="gcp-eu-1" SLURM=1 CPUS=12 GPUS=4 MEM=20GB TIME="04:00:00" bash ./run/run_any.sh python ./train.py --config ./configs/config_world_dreamer_retro.yaml --description your_model_one_line_description
#
# use SKIP_INSTALL=1 if you are sure the environment is set up and up-to-date already on that node to directly run.
# 
# 3) Reserve interactively gpus on hala:
# srun --nodelist=hala --gres=gpu:a6000:1 --cpus-per-task=12 --ntasks=1 --mem=40GB --time=04:00:00 --pty /bin/bash
# then you can run locally:
# SLURM=0 ENV_NAME="ivg" bash ./run/run_any.sh python {your_script.py} {--your script --parameters}
# or even directly without run_any


CURRENT_DPATH=$(dirname "$(readlink -f "$0")")
source $CURRENT_DPATH/../ivg.env

SLURM=${SLURM:-1}
SALLOC=${SALLOC:-0}
CPUS=${CPUS:-12}
GPUS=${GPUS:-1}
GPU_MODEL=${GPU_MODEL:-"a100-40g"}
MEM=${MEM:-50GB}
TIME=${TIME:-"04:00:00"}
export ENV_NAME=${ENV_NAME:-$IVG_ENV_NAME}
export SKIP_INSTALL=${SKIP_INSTALL:-0}

if [ -n "$WANDB_MODE" ]; then
    export WANDB_MODE
fi

#There must be a node set if SLURM is enabled
# if [ -z "$NODE" ] && [ "$SLURM" -eq 1 ] && [ "$SALLOC" -eq 0 ]; then
#     #show an error message
#     echo "Error: SLURM is enabled, but NODE is unset. Please set NODE and try again."
#     exit 1
# fi

if [ -n "$NODE" ]; then
    if [[ "$NODE" == gcpl4* ]]; then
        GPU_MODEL="l4-24g"
    elif [[ "$NODE" == gcp* ]]; then
        GPU_MODEL="a100-40g"
    elif [[ "$NODE" == hala* ]]; then
        GPU_MODEL="a6000"
    fi
fi

if [ "$GPU_MODEL" = "a6000" ]; then
    export NCCL_P2P_DISABLE=1
fi

if [ "$SLURM" -eq 0 ]; then
    export SKIP_INSTALL=1
fi

if [ "$SKIP_INSTALL" -eq 0 ]; then
    bash $IVG_SCRIPTS_DPATH/install_ivg_data_upload.sh
else
    echo "Skipping data and env installation!!!"
fi

NODELIST_PARAM=""
if [ -n "$NODE" ] && [ "$NODE" != "any" ]; then
    NODELIST_PARAM="--nodelist=$NODE"
fi

export PYTHONPATH=$PYTHONPATH:$IVG_PROJECT_DPATH:$PVG_PROJECT_DPATH:$VICREG_PROJECT_DPATH
export PYTHONPATH=$PYTHONPATH:$DATAGEN_PROJECT_DPATH

# exit 0
if [ "$SLURM" -eq 1 ]; then
    echo "Starting a SLURM job on node $NODE with $CPUS CPUs, $GPUS GPUs, $MEM memory, and $TIME time"
    if [ "$SALLOC" -ge 1 ]; then
        if [ "$SALLOC" -eq 1 ]; then
            SALLOC_FPATH=~/.salloc_jobs
        else
            SALLOC_FPATH=~/.salloc_jobs_$SALLOC
        fi
        SALLOC_JOB_ID=$(cat $SALLOC_FPATH)
        srun --jobid=$SALLOC_JOB_ID --gres=gpu:$GPU_MODEL:$GPUS --ntasks=1 --gpus-per-task=$GPU_MODEL:$GPUS --overlap bash $IVG_SCRIPTS_DPATH/slurm_gpu.sh "$@" # --output=log/%j.out --error=log/%j.out
    else    
        srun $NODELIST_PARAM --nodes=1 --cpus-per-task=$CPUS --mem=$MEM --gres=gpu:$GPU_MODEL:$GPUS --ntasks=1 --gpus-per-task=$GPU_MODEL:$GPUS --time=$TIME bash $IVG_SCRIPTS_DPATH/slurm_gpu.sh "$@" # --output=log/%j.out --error=log/%j.out
    fi
    
    # logwatch
else
    bash $IVG_SCRIPTS_DPATH/slurm_gpu.sh "$@"
fi
