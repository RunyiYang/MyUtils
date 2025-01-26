##root/scripts/install_conda_env.sh
#installs a conda env by a given name. it is expected that the env will be built or updated
#from an already existing file: root/{ENV_NAME}_env.yaml

#set up env variables
if [ -z "$SLURM_JOB_ID" ]; then
    CURRENT_DPATH=$(dirname "$(readlink -f "$0")")
else
    source ~/.bashrc
    #show an error
    if [ -z "$IVG_ROOT_DPATH" ]; then
        echo "Error: SLURM is enabled, but IVG_ROOT_DPATH is unset. Please set IVG_ROOT_DPATH and try again."
        exit 1
    fi
    CURRENT_DPATH=$IVG_ROOT_DPATH/scripts
fi
source $CURRENT_DPATH/setup_shell_env.sh
ENV_NAME=$1
PROJECT_DPATH=$2

# conda shell init --shell bash --root-prefix=$ENV_ROOT_DPATH

ENV_EXISTS=$(conda env list | grep -w $ENV_NAME)
if [ -z "$ENV_EXISTS" ]; then
    echo "Creating environment '$ENV_NAME'..."
    conda env create -f $PROJECT_DPATH/${ENV_NAME}_env.yml -y
    conda init
    source ~/.bashrc
else
    echo "Environment '$ENV_NAME' already exists. Updating..."
    conda env update --name $ENV_NAME -f $PROJECT_DPATH/${ENV_NAME}_env.yml --prune
    if [ $? -ne 0 ]; then
        echo "Updating environment failed. Removing and recreating..."
        conda env remove --name $ENV_NAME -y
        conda env create -f $PROJECT_DPATH/${ENV_NAME}_env.yml -y
        conda init
        source ~/.bashrc
    fi
fi
