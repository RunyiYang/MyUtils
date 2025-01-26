##root/scripts/install_miniconda.sh
#installs miniconda on a predetermined path on scratch

USERNAME=`whoami`
MINICONDA_DPATH=${MINICONDA_DPATH:-"/scratch/$USERNAME/tools/miniconda3"}
TOOLS_DIR=`dirname $MINICONDA_DPATH`

if [ ! -d "$MINICONDA_DPATH" ]; then
    echo "Installing miniconda3"
    mkdir -p $TOOLS_DIR
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $TOOLS_DIR/miniconda3.sh
    bash $TOOLS_DIR/miniconda3.sh -b -p $MINICONDA_DPATH
    rm $TOOLS_DIR/miniconda3.sh
    
    $MINICONDA_DPATH/bin/conda update --all
    $MINICONDA_DPATH/bin/conda config --set solver libmamba
fi

#conda activate the base environment
$MINICONDA_DPATH/bin/activate
#conda activate the base environment without it being in PATH yet
$MINICONDA_DPATH/bin/conda init
