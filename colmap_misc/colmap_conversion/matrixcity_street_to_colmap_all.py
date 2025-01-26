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


def gl2world_to_cv2world(gl2world):
    cv2gl = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    cv2world = gl2world @ cv2gl

    return cv2world




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
    # parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index')
    parser.add_argument('--end_idx', type=int, default=1e8, help='End index')
    parser.add_argument('--merge', action='store_true', help='Merge all the images', default=False)
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()


    input_dir = args.input_dir
    VIEW_NAME = "street"
    OUT_NAME = "steet_blocks_all_th50"
    STREET_NAMEs = ["small_city_road_down",  "small_city_road_horizon",  "small_city_road_outside",  "small_city_road_vertical"]
    pointcloud_path = '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city_pointcloud/point_cloud_ds20/street/Block_all.ply'
    task_split = ['train', 'test']


    #  '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city_pointcloud/point_cloud_ds20/street/Block_all.ply'
    #  '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/oct_gs/matrix_city_init.ply'
    # '/data/work2-gcp-europe-west4-a/qimaqi/datasets/MatrixCity/small_city_pointcloud/point_cloud_ds20/street/Block_all.ply' # Block_A Block_all
    output_dir = os.path.join(input_dir, OUT_NAME, 'camera_calibration' ,'rectified')


    # we read camera parameters from block all
    with open(os.path.join(input_dir, VIEW_NAME , "pose",  'block_A', f'transforms_train.json'), "r") as f:
        meta_file = json.load(f)
    high_angle_x = meta_file['camera_angle_x']
    fl_x = meta_file['fl_x']
    fl_y = meta_file['fl_y']
    # high_cx = meta_file['cx']
    w = float(1000)
    h = float(1000)
    # fl_x = float(.5 * w / np.tan(.5 * high_angle_x))
    # fl_y = fl_x
    cx = w / 2
    cy = h / 2
    # high_cy = meta_file['cy']
    high_w = meta_file['w']
    high_h = meta_file['h']
    print("fl_x", fl_x, "fl_y", fl_y, "cx", cx, "cy", cy, "w", w, "h", h)


    c2ws_all = []
    imgs_path_all = []
    depth_path_all = []

    # frames_block_A = meta_file['frames']
    for task_i in task_split:
        for street_name in STREET_NAMEs:
            if task_i == 'test':
                street_name = street_name + '_test'
            
            transform_path_sub = os.path.join(input_dir, VIEW_NAME , task_i,  street_name, f'transforms.json')
            
            with open(transform_path_sub, "r") as f:
                meta_block_i = json.load(f)   
                meta_block_frame_i = meta_block_i['frames']
                for count_i, frame in enumerate(meta_block_frame_i):
                    img_idx = frame["frame_index"]
                    img_path_abs = os.path.join(input_dir, VIEW_NAME , task_i,  street_name, str(img_idx).zfill(4)+'.png')
                    depth_path_abs = os.path.join(input_dir.replace('small_city', 'small_city_depth'), VIEW_NAME , task_i,  street_name+'_depth', str(img_idx).zfill(4)+'.exr')
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
                        raise ValueError(f"img_path_abs or depth path abs not exists, {img_path_abs}, {depth_path_abs}")


    # with all images and depth, start preparing colmap
    # sort based only on basename
    # imgs_path_all = sorted(imgs_path_all, key=lambda x: int(os.path.basename(x)[:-4]))
    # depth_path_all = sorted(depth_path_all, key=lambda x: int(os.path.basename(x)[:-4]))
    # imgs_path_all = sorted(imgs_path_all)
    # depth_path_all = sorted(depth_path_all)
    c2ws_all = np.stack(c2ws_all) #[B,4,4]
    centers = c2ws_all[:, :3, 3]

    print("centers x min", np.min(centers[:,0]), "centers x max", np.max(centers[:,0]))
    print("centers y min", np.min(centers[:,1]), "centers y max", np.max(centers[:,1]))
    print("centers z min", np.min(centers[:,2]), "centers z max", np.max(centers[:,2]))
    # centers x min -851.5166015625001 centers x max -120.55117034912108
    # centers y min -546.9083251953125 centers y max -0.22081592679023743
    # centers z min 3.0 centers z max 3.0
    # raise ValueError("stop here")
    # self.points3d['XYZ'] x range -1490.6496047973633 1363.3264541625977
    # self.points3d['XYZ'] y range -1304.3548583984375 1228.7542343139648
    # self.points3d['XYZ'] z range -3.401993215084076 502.8411388397217

    converter = ImageDepth2Colmap(imgs_path_all, depth_path_all, c2ws_all ,pointcloud_path, {'model': 'PINHOLE', 'width': int(high_w), 'height': int(high_h), 'params': [fl_x, fl_y, cx, cy]}, output_dir)


    converter.collect_files()
    print("finish collect_files")
    # save cameras.txt
    converter.save_cameras_txt()
    converter.init_globl_points3d()
    
    if not args.merge:
        converter.save_images_txt(depth_max=50, depth_threshold=0.1, debug=False,tmp_save=False, start_idx=args.start_idx, end_idx=args.end_idx)
    else:
        converter.merge_images_points_txt() 



if __name__ == '__main__':
    main()