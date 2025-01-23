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
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from colmap_conversion_utils import Image2Colmap

def gl2world_to_cv2world(gl2world):
    cv2gl = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    cv2world = gl2world @ cv2gl

    return cv2world


def arg_parser():
    parser = argparse.ArgumentParser(description='MatrixCity to Colmap')
    parser.add_argument('--input_dir', type=str, default='/srv/beegfs-benderdata/scratch/qimaqi_data/data/iccv_2025/datasets/Dataset/scripts/DL3DV-10K/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f', help='Input directory')
    parser.add_argument('--rescale', type=int, default=4)
    # parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    input_dir = args.input_dir
    rescale = args.rescale


    output_dir = os.path.join(input_dir, 'colmap', 'sparse', 'known')
    os.makedirs(output_dir, exist_ok=True)


    # we read camera parameters from block all
    with open(os.path.join(input_dir, f'transforms.json'), "r") as f:
        meta_file = json.load(f)

    h = meta_file['h'] / rescale
    w = meta_file['w'] / rescale

    fl_x = meta_file['fl_x'] / rescale
    fl_y = meta_file['fl_y'] / rescale

    cx = meta_file['cx'] / rescale
    cy = meta_file['cy'] / rescale

    k1 = meta_file['k1']
    k2 = meta_file['k2']
    p1 = meta_file['p1']
    p2 = meta_file['p2']


    print("fl_x", fl_x, "fl_y", fl_y, "cx", cx, "cy", cy, "w", w, "h", h)


    c2ws_all = []
    imgs_path_all = []

    frames_org = meta_file['frames']
    for count_i, frame in enumerate(frames_org):
        file_path = frame["file_path"]
        if rescale != 1:
            file_path = file_path.replace("images", f"images_{rescale}")

        # check camera coordainte system, assume to be opengl
        c2w=np.array(frame["transform_matrix"])

        # rot_mat = c2w[:3, :3]
        # # check rot mat is valid or not
        # composed_mat = rot_mat @ rot_mat.T
        # # if not np.allclose(composed_mat, np.eye(3)):
        # #     c2w[:3,:3]*=100 # bug of data
        # assert np.sum(np.abs(composed_mat - np.eye(3))) <1e-5, f"composed_mat {composed_mat}"
        # c2w = opengl_to_opencv(c2w)
        c2w = gl2world_to_cv2world(c2w)
        c2ws_all.append(c2w.tolist()) 


        img_path_abs = os.path.join(input_dir, file_path)
  
        if os.path.exists(img_path_abs):
            imgs_path_all.append(img_path_abs)
            # debug print
            if count_i % 100 == 0:
                # for debug use
                print("add img_path_abs", img_path_abs)

        else:
            raise ValueError(f"img_path_abs {img_path_abs} or depth_path_asb {depth_path_asb} not exists")

    c2ws_all = np.stack(c2ws_all) #[B,4,4]
    # np.save('./debug_pose.npy', c2ws_all)
    # raise ValueError("stop here")
    c2ws_all = np.stack(c2ws_all) #[B,4,4]
    centers = c2ws_all[:, :3, 3]

    # print("centers x min", np.min(centers[:,0]), "centers x max", np.max(centers[:,0]))
    # print("centers y min", np.min(centers[:,1]), "centers y max", np.max(centers[:,1]))
    # print("centers z min", np.min(centers[:,2]), "centers z max", np.max(centers[:,2]))


    converter = Image2Colmap(imgs_path_all, c2ws_all, {'model': 'OPENCV', 'width': int(w), 'height': int(h), 'params': [fl_x, fl_y, cx, cy, k1, k2, p1, p2]}, output_dir)


    print("finish collect_files")
    # save cameras.txt
    converter.save_cameras_txt()
    converter.save_images_txt()




if __name__ == '__main__':
    main()