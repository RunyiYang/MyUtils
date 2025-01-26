##root/scripts/setup_shell_env.sh
# A utility script to load the variables in ivg.env

source ~/.bashrc
#check if IVG_ENV_SET eists
if [ -z "$IVG_ENV_SET" ]; then
    #check if we are using slurm
    if [ -z "$SLURM_JOB_ID" ]; then
        #set up env variables
        ROOT_DPATH=$(dirname "$(dirname "$(readlink -f "$0")")")
        source $ROOT_DPATH/ivg.env
    else
        #show an error
        if [ -z "$IVG_ROOT_DPATH" ]; then
            echo "Error: SLURM is enabled, but IVG_ROOT_DPATH is unset. Please set IVG_ROOT_DPATH and try again."
            exit 1
        fi
        source $IVG_ROOT_DPATH/ivg.env
    fi
fi
