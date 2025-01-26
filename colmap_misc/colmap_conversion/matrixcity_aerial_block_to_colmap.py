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
from read_write_model import read_images_binary,write_images_binary, Image
import cv2 
from PIL import Image as ImagePIL
import trimesh 
from matplotlib import pyplot as plt
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

from colmap_conversion_utils import ImageDepth2Colmap

def depths_exr_to_inv_depths_png(depths_exr_path, inv_depths_png_path):
    depth_exr, _ = load_depth(depths_exr_path)
    inv_depth = (1.0 * 2**16) / depth_exr
    # change to float 32
    inv_depth = inv_depth.astype(np.float32)
    # times anther 2**16 then save
    # save to png file
    cv2.imwrite(inv_depths_png_path, inv_depth)


def gl2world_to_cv2world(gl2world):
    cv2gl = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    cv2world = gl2world @ cv2gl

    return cv2world

def load_depth(depth_path, is_float16=True):
    image = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    if is_float16==True:
        invalid_mask=(image==65504)
    else:
        invalid_mask=None
    image = image / 100 # cm -> m
    return image, invalid_mask

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K



def save_cameras_txt(cameras, output_path):
    """
    Save camera intrinsics to cameras.txt in COLMAP format.
    
    Args:
    - cameras (list): List of dictionaries with camera model parameters.
    - output_path (str): Path to save the cameras.txt file.
    """
    camera_file_path = os.path.join(output_path, 'cameras.txt')
    with open(camera_file_path, 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        
        for i, camera_params in enumerate(cameras):
            camera_id = i + 1
            model_id = camera_params['model']
            width = camera_params['width']
            height = camera_params['height']
            params = ' '.join(map(str, camera_params['params']))  # fx, fy, cx, cy, distortion params
            f.write(f'{camera_id} {model_id} {width} {height} {params}\n')
    
    print(f"Saved cameras to {camera_file_path}")




def matrix_to_quaternion_and_translation(matrix):
    """Convert 4x4 camera-to-world matrix to quaternion and translation."""
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Convert rotation matrix to quaternion (x, y, z, w)
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    # change x y z w to w x y z
    quaternion = np.roll(quaternion, 1)

    return quaternion, translation


def arg_parser():
    parser = argparse.ArgumentParser(description='MatrixCity to Colmap')
    parser.add_argument('--input_dir', type=str, default='/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city/', help='Input directory')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=1e8, help='End index')
    parser.add_argument('--merge', action='store_true', help='Merge all the images', default=False)
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    input_dir = args.input_dir
    HIGH_NAME = "aerial"
    OUT_NAME = "aerial_all_blocks_debug"
    output_dir = os.path.join(input_dir, OUT_NAME, 'camera_calibration' ,'rectified')
    pointcloud_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city_pointcloud/point_cloud_ds20/aerial/Block_all.ply' # Block_A Block_all

    # debug first, let's try first with only block_A
    AERIAL_BLOCK_NAMEs = ['block_all'] # ["block_all"] #['block_A']
    task_split = ['train', 'test']


    # we read camera parameters from block all
    with open(os.path.join(input_dir, HIGH_NAME , "pose",  'block_all', f'transforms_train.json'), "r") as f:
        meta_high_train = json.load(f)
    high_angle_x = meta_high_train['camera_angle_x']
    high_w = float(1920)
    high_h = float(1080)
    fl_x = float(.5 * high_w / np.tan(.5 * high_angle_x))
    fl_y = fl_x
    cx = high_w / 2
    cy = high_h / 2


    c2ws_all = []
    imgs_path_all = []
    depth_path_all = []
    for aerial_block_name in AERIAL_BLOCK_NAMEs:
        for task in task_split:
            if aerial_block_name == 'block_A':
                aerial_sub_blocks_names = ['block_1', 'block_2']
            elif aerial_block_name == 'block_B' or aerial_block_name == 'block_C':
                aerial_sub_blocks_names = ['block_3','block_4','block_5','block_6','block_7','block_8']
            elif aerial_block_name == 'block_D':
                aerial_sub_blocks_names = ['block_9']
            elif aerial_block_name == 'block_E':
                aerial_sub_blocks_names = ['block_10']
            elif aerial_block_name == 'block_all':
                aerial_sub_blocks_names = ['block_1', 'block_2', 'block_3','block_4','block_5','block_6','block_7','block_8','block_9','block_10']


            if task == 'test':
                aerial_sub_blocks_names = [aerial_sub_blocks_name + "_test" for aerial_sub_blocks_name in aerial_sub_blocks_names]

            # go through each block folder
            for aerial_sub_blocks_name in aerial_sub_blocks_names:
                transform_path_sub = os.path.join(input_dir, HIGH_NAME , task,  aerial_sub_blocks_name, f'transforms.json')
                with open(transform_path_sub, "r") as f:
                    meta_block_i = json.load(f)
                    meta_block_frame_i = meta_block_i['frames']
                    for count_i, frame in enumerate(meta_block_frame_i):
                        img_idx = frame["frame_index"]
                        img_path_abs = os.path.join(input_dir, HIGH_NAME , task,  aerial_sub_blocks_name, str(img_idx).zfill(4)+'.png')
                        depth_path_abs = os.path.join(input_dir.replace('small_city', 'small_city_depth'), HIGH_NAME , task,  aerial_sub_blocks_name+'_depth', str(img_idx).zfill(4)+'.exr')
                        if os.path.exists(img_path_abs) and os.path.exists(depth_path_abs):
                            imgs_path_all.append(img_path_abs)
                            depth_path_all.append(depth_path_abs)
                            c2w=np.array(frame["rot_mat"])
                            # check unit of distance (m or 100m)
                            c2w[3,3]=1
                            rot_mat = c2w[:3, :3]
                            # check rot mat is valid or not
                            composed_mat = rot_mat*rot_mat.T
                            if not np.allclose(composed_mat, np.eye(3)):
                                c2w[:3,:3]*=100 # bug of data
                            assert not np.allclose(composed_mat, np.eye(3)), f"composed_mat {composed_mat}"
                            c2w = gl2world_to_cv2world(c2w)
                            c2ws_all.append(c2w.tolist()) 
                            if count_i % 100 == 0:
                                # for debug use
                                print("add img_path_abs", img_path_abs)
                                print("add depth_path_abs", depth_path_abs)
                        else:
                            raise ValueError("img_path_abs or depth path abs not exists")

    c2ws_all = np.stack(c2ws_all) #[B,4,4]

    centers = c2ws_all[:, :3, 3]

    print("centers x min", np.min(centers[:,0]), "centers x max", np.max(centers[:,0]))
    print("centers y min", np.min(centers[:,1]), "centers y max", np.max(centers[:,1]))
    print("centers z min", np.min(centers[:,2]), "centers z max", np.max(centers[:,2]))
    # block A
    # camera pos x -1000.0 -120.0
    # camera pos y -630.0 0.0
    # camera pos z 150.0 150.0
    # self.points3d['XYZ'] x range -1420.857810974121 458.61048698425293
    # self.points3d['XYZ'] y range -1199.0580558776855 583.8757038116455
    # self.points3d['XYZ'] z range -3.364799916744232 502.5200843811035

    # stop here
    # raise ValueError("stop here")

    converter = ImageDepth2Colmap(imgs_path_all, depth_path_all, c2ws_all ,pointcloud_path, {'model': 'PINHOLE', 'width': int(high_w), 'height': int(high_h), 'params': [fl_x, fl_y, cx, cy]}, output_dir)
    converter.collect_files()
    # save cameras.txt
    converter.save_cameras_txt()
    converter.init_globl_points3d()
    
    # converter.save_images_txt(depth_threshold=1, depth_max=500, debug=False)
    if not args.merge:
        converter.save_images_txt(depth_max=500, depth_threshold=1, debug=False,tmp_save=False, start_idx=args.start_idx, end_idx=args.end_idx)
    else:
        converter.merge_images_points_txt() 


if __name__ == '__main__':
    main()