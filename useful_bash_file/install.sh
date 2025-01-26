##root/scripts/install.sh
#used in slurm_gpu.sh to set up conda and potentially more setup steps can be added here
#this can be skipped by setting SKIP_INSTALL=1 env when calling run_any.sh

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
echo "Installing Miniconda..."
bash $IVG_SCRIPTS_DPATH/install_miniconda.sh
echo "Installing Conda Environment..."
bash $IVG_SCRIPTS_DPATH/install_conda_env.sh $ENV_NAME $(pwd)
# echo "Starting Data Download..."
# bash $IVG_SCRIPTS_DPATH/install_ivg_data_download.sh

echo "Done."