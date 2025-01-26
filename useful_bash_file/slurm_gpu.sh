##root/scripts/slurm_gpu.sh
# This script is ran on the remote node. It is responsible for setting things up (running the install scripts)
# and also running the command. run_any makes sure to run this script on the remote node.

#!/bin/bash
#SBATCH --nodelist=[] # Appoint specific GPU nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --output=log/%j.out
#SBATCH -e log/%j.out
#SBATCH --constraint="type-gpu"
#SBATCH --time=04:00:00

if [ -z "$IVG_PROJECT_DPATH" ]; then
    #show an error message
    echo "Error: ROOT_DPATH is unset. Please set ROOT_DPATH and try again."
    exit 1
fi

source ~/.bashrc
source $IVG_PROJECT_DPATH/ivg.env
if [ "$SKIP_INSTALL" -eq 0 ]; then
    #if the sh path exists
    if [ -f $IVG_SCRIPTS_DPATH/install_$ENV_NAME.sh ]; then
        bash $IVG_SCRIPTS_DPATH/install_$ENV_NAME.sh
    else
        bash $IVG_SCRIPTS_DPATH/install.sh
    fi
fi
# mamba activate $PVG_ENV_NAME
conda activate $ENV_NAME

# Run the task on each GPU node separately
PYAV_LOGGING=off TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_ASYNC_ERROR_HANDLING=1 "$@"
