# licence : MIT
# Authors: Qi Ma
# Date: 2024-09
# Contact: qimaqi@ethz.ch
# colmap model_converter --input_path=/work/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap --output_path=/work/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0 --output_type=BIN

# usage: this code change scenes in matrix city both aerial and street, (both train and test) to colmap format
# python matrixcity_to_colmap.py --input_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city --output_dir /data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/
# 
# python preprocess/auto_reorient_npts.py --input_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/rectified/sparse/0 --output_path=/data/work-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/aerial_street_colmap/camera_calibration/aligned/sparse/0  --upscale=1

import os
import argparse
import json 
import numpy as np
import PIL

import numpy as np
import os
from scipy.spatial.transform import Rotation

# 
import numpy as np
from scipy.spatial.transform import Rotation
import os
import shutil
# from read_write_model import read_images_binary,write_images_binary, Image
import cv2 
from PIL import Image as ImagePIL
import trimesh 
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch 
from colmap_conversion_utils import Image2Colmap

def arg_parser():
    parser = argparse.ArgumentParser(description='MatrixCity to Colmap')
    parser.add_argument('--root_dir', type=str, default='/data/work2-gcp-europe-west4-a/GSWorld/DL3DV/', help='Root directory')
    parser.add_argument('--input_dir', type=str, default='11K/', help='Input directory')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    # parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    return args


def run_colmap():
    args = arg_parser()
    dataset_dir = os.path.join(args.root_dir, args.input_dir)
    input_dir = args.input_dir
    sequences = os.listdir(dataset_dir) 
    # only save folder
    # sequences = [seq for seq in sequences if os.path.isdir(os.path.join(input_dir, seq))]
    sequences = sorted(sequences)
    print("Total Sequences: ", len(sequences))
    if args.end_idx != -1:
        if args.end_idx > len(sequences):
            args.end_idx = len(sequences)
        sequences = sequences[args.start_idx:args.end_idx]
    else:
        sequences = sequences[args.start_idx:]

    print("Sub Sequences: ", sequences)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()  # Total number of GPUs available
        available_gpus = [i for i in range(num_gpus)]  # List of GPU indices
        print(f"Available GPU indices: {available_gpus}")
        gpu_idx = available_gpus[0]
    else:
        print("No GPUs are available.")
        
    for seq_i in tqdm(sequences):
        seq_dir = os.path.join(dataset_dir, seq_i)
        print("=====================================")
        print(f"Processing {seq_i}")
        fuse_ply_path = os.path.join(seq_dir, 'fused.ply')

        if os.path.exists(fuse_ply_path):
            print(f"Already processed: {fuse_ply_path}")
            continue

        shutil.rmtree(os.path.join(seq_dir, 'colmap'), ignore_errors=True)


        
        # move work dir to /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/colmap_misc/colmap_pointcloud
        os.chdir('/home/runyi_yang/outdoor/MyUtils/colmap_misc/colmap_pointcloud')
        os.system(f"python dl3dv_10k_to_colmap.py --input_dir {seq_dir}") 
        assert os.path.exists(os.path.join(seq_dir, 'colmap', 'sparse', 'known', 'points3D.txt')), f"Fuse ply not created: {os.path.join(seq_dir, 'fused.ply')}"
        # we have known model, go through dense colmap process
        # chage work dir to seq_dir 
        os.chdir(seq_dir)
        colmap_image_path = os.path.join(seq_dir, 'images_4')
        colmap_database_path = os.path.join(seq_dir, 'colmap', 'database.db')
        os.makedirs(os.path.join(seq_dir, 'colmap'), exist_ok=True)
        docker_image_path = colmap_image_path.replace(dataset_dir, "/data/")
        docker_database_path = colmap_database_path.replace(dataset_dir, "/data/")
        os.system(f"slurm-podman-run -it --rm -v {dataset_dir}:/data colmap-cuda-interactive colmap feature_extractor --database_path {docker_database_path} --image_path {docker_image_path}")

        # check if the database is created
        assert os.path.exists(colmap_database_path), f"Database not created: {colmap_database_path}"
        
        os.system(f"slurm-podman-run -it --rm -v {dataset_dir}:/data colmap-cuda-interactive colmap exhaustive_matcher --database_path {docker_database_path} --SiftMatching.gpu_index={gpu_idx}")

        triangulated_path = os.path.join(seq_dir, 'colmap', 'triangulated', 'sparse', 'model')
        os.makedirs(triangulated_path, exist_ok=True)
        docker_triangular_path = triangulated_path.replace(dataset_dir, "/data/")
        docker_seq_dir = seq_dir.replace(dataset_dir, "/data/")
        os.system(f"slurm-podman-run -it --rm -v {dataset_dir}:/data colmap-cuda-interactive colmap point_triangulator --database_path {docker_database_path} --image_path {docker_image_path} --output_path {docker_triangular_path} --input_path {os.path.join(docker_seq_dir, 'colmap', 'sparse', 'known')}")

        assert os.path.exists(os.path.join(triangulated_path,'points3D.bin')), f"Triangulated points not created: {os.path.join(triangulated_path,'points3D.bin')}"

        dense_output_path = os.path.join(seq_dir, 'colmap', 'dense')
        docker_dense_output_path = dense_output_path.replace(dataset_dir, "/data/")
        os.makedirs(dense_output_path, exist_ok=True)
        docker_fuse_ply_path = fuse_ply_path.replace(dataset_dir, "/data/")
        

        os.system(f"slurm-podman-run -it --rm -v {dataset_dir}:/data colmap-cuda-interactive colmap image_undistorter --input_path {docker_triangular_path} --output_path {docker_dense_output_path} --image_path {docker_image_path}")

        os.system(f"slurm-podman-run -it --rm -v {dataset_dir}:/data colmap-cuda-interactive colmap patch_match_stereo --workspace_path {docker_dense_output_path} --PatchMatchStereo.gpu_index={gpu_idx}")

        os.system(f"slurm-podman-run -it --rm -v {dataset_dir}:/data colmap-cuda-interactive colmap stereo_fusion --workspace_path {docker_dense_output_path} --output_path {docker_fuse_ply_path}")

        assert os.path.exists(fuse_ply_path), f"Fuse ply not created: {fuse_ply_path}"
        
        # clean the dense folder 
        # too large 
        shutil.rmtree(os.path.join(seq_dir, 'colmap'))



if __name__ == '__main__':
    run_colmap()