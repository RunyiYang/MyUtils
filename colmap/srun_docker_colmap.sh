#!/bin/bash
#SBATCH --job-name=Trydocker
#SBATCH --nodelist=gcpl4-eu-1
#SBATCH --partition=debug
#SBATCH --gpus=l4-24g:1          # Specify the partition
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # Number of tasks (total)
#SBATCH --cpus-per-task=16       # Number of CPU cores (threads) per task
#SBATCH --mem-per-cpu=16G        # Memory limit per CPU core (there is no --mem-per-task)
#SBATCH --time=4:00:00        # Job timeout
#SBATCH --output=colmap_docker.log      # Redirect stdout to a log file
#SBATCH --error=colmap_docker.error     # Redirect stderr to a separate error log file
#SBATCH --mail-type=ALL         # Send updates via email

# podman build -t colmap-cuda-interactive:latest .

slurm-podman-run --rm -it --gpu\
    -v /home/runyi_yang/SGSLAM:/data \
    colmap-cuda:latest colmap automatic_reconstructor \
    --workspace_path /data/MyUtils/colmap_replica_try \
    --image_path /data/SGSLAM/datasets/replica/office0/images