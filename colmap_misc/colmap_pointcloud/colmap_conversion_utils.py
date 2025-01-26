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
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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



def opengl_to_opencv(w2gl):
    """
    Change opengl to opencv
    """
    # world to camera opengl, then opengl to opencv
    gl2cv = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
    w2cv = gl2cv @ w2gl

    return w2cv



def render_3d_world_to_camera_opencv(points3d, w2c, intrinsics_params, depth_max=100):
    # using opencv function to do it
    # change w2c to rvec and tvec
    # w2c = opengl_to_opencv(w2c)
    w2c = w2c[:3, :]
    points3d_xyz = points3d['XYZ']
    points3d_id = points3d['POINT3D_ID']
    points3d_xyz = points3d_xyz.T  # 3xN
    points3d_xyz = np.vstack((points3d_xyz, np.ones((1, points3d_xyz.shape[1]))))   # 4xN
    # step2 change to camera coordinate
    points3d_xyz = w2c @ points3d_xyz  # 3xN
    points3d_xyz = points3d_xyz[:3, :] # 3xN

    # select based on depth > 0 and < depth_max

    points3d_xyz = points3d_xyz.T  # Nx3
    z_mask = (points3d_xyz[:, 2] > 0) & (points3d_xyz[:, 2] < depth_max)
    points3d_xyz = points3d_xyz[z_mask]
    points3d_id = points3d_id[z_mask]


    # save this Nx3 as ply file for visualization using trimesh
    # cloud = trimesh.PointCloud(points3d_xyz)

    # # Save the colored point cloud as a .ply file
    # cloud.export('./street_debug/points_cam_coord.ply')
        

    # rvec = cv2.Rodrigues(w2c[:3, :3])[0]
    # tvec = w2c[:3, -1]
    rvec = cv2.Rodrigues(np.eye(3))[0]
    tvec = np.zeros(3)

    H = intrinsics_params['height']
    W = intrinsics_params['width']
    intrinsics = as_intrinsics_matrix(intrinsics_params['params'])

    # No distortion coefficients (assuming none)
    dist_coeffs = None #np.zeros(4)

    # Project 3D points to 2D
    points2d, _ = cv2.projectPoints(points3d_xyz, rvec, tvec, intrinsics, dist_coeffs)

    depth_map = np.full((H, W), np.inf)
    mask_map = np.zeros((H, W), dtype=np.uint8)
    used_points3d_id = np.zeros((H, W), dtype=np.int32) # each coordinate save one id

    # Iterate over the projected points and update the depth map
    for i, (point, img_pt) in enumerate(zip(points3d_xyz, points2d)):
        x, y = int(img_pt[0][0]), int(img_pt[0][1])  # 2D image coordinates
        z = point[-1]  # Depth (z-value in the original 3D point)

        if z > 0 and z < depth_max:
            # Make sure the point is within the image bounds
            if 0 <= x < W and 0 <= y < H:
                # Update depth map with the minimum depth (in case of overlapping points)
                if z < depth_map[y, x]:
                    depth_map[y, x] = z
                    used_points3d_id[y, x] = points3d_id[i]
                    # print("depth_map[y, x]", depth_map[y, x])
                mask_map[y, x] = 1

    
    return depth_map, mask_map, used_points3d_id


class Image2Colmap():
    def __init__(self, image_paths, poses, intrinsics, output_path):
        self.image_paths = image_paths
        self.intrinsics = intrinsics
        self.output_path = output_path
        self.poses = poses

    # def collect_files(self, overwrite=False):
    #     # copy images and depth to output_path/rectified/images and output_path/rectified/depth
    #     os.makedirs(self.output_path, exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'images'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'depths_exr'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'depths'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'sparse', 'known'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'sparse', '0'), exist_ok=True)

    #     # also save a new transform.json
    #     copy_map = {}
    #     new_transforms_path = os.path.join(self.output_path, 'sparse', 'known' , 'transforms.json')
    #     new_transforms = {}
    #     new_transforms['train'] = {}
    #     new_transforms['test'] = {}
    #     new_transforms['train_dense'] = {}

    #     for i, (img_path, depth_path) in tqdm(enumerate(zip(self.image_paths, self.depth_paths)),total=len(self.image_paths) ):
    #         img_target_path = os.path.join(self.output_path, 'images', f"{str(i).zfill(8)}.png")
    #         depth_target_path = os.path.join(self.output_path, 'depths_exr', f"{str(i).zfill(8)}.exr")
    #         inv_depth_target_path = os.path.join(self.output_path, 'depths', f"{str(i).zfill(8)}.png")
   
    #         if not overwrite:
    #             if not os.path.exists(img_target_path):
    #                 shutil.copy(img_path, img_target_path)
    #             if not os.path.exists(depth_target_path):
    #                 shutil.copy(depth_path, depth_target_path)
    #         else:
    #             shutil.copy(img_path, img_target_path)
    #             shutil.copy(depth_path, depth_target_path)
    #         self.image_paths_out.append(img_target_path)
    #         self.depth_paths_out.append(depth_target_path)
    #         split = img_path.split("/")[-3]
    #         block_name = img_path.split("/")[-2]
    #         data_dict = {str(i).zfill(8): self.poses[i].tolist()}
    #         if block_name not in new_transforms[split]:
    #             new_transforms[split][block_name] = [data_dict]
    #         else:
    #             new_transforms[split][block_name].append(data_dict)
    #         # new_transforms[split].append(data_dict)
    #         if not os.path.exists(inv_depth_target_path):
    #             depths_exr_to_inv_depths_png(depth_target_path, inv_depth_target_path)

    #     with open(new_transforms_path, 'w') as f:
    #         json.dump(new_transforms, f, indent=4)


    def save_cameras_txt(self):
        """
        Save camera intrinsics to cameras.txt in COLMAP format.
        
        Args:
        - cameras (list): List of dictionaries with camera model parameters.
        - output_path (str): Path to save the cameras.txt file.
        """
        camera_file_path = os.path.join(self.output_path, 'cameras.txt')

        if os.path.exists(camera_file_path):
            os.remove(camera_file_path)

        with open(camera_file_path, 'w') as f:
            f.write('# Camera list with one line of data per camera:\n')
            f.write('# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
            
            camera_id = 1
            model_id = self.intrinsics['model']
            width = self.intrinsics['width']
            height = self.intrinsics['height']
            params = ' '.join(map(str, self.intrinsics['params']))  
            f.write(f'{camera_id} {model_id} {width} {height} {params}\n')
        print(f"Saved cameras to {camera_file_path}")

 
    def save_images_txt(self):
        image_file_path = os.path.join(self.output_path , 'images.txt')

        with open(image_file_path, 'w') as f:
            f.write('# Image list with one line of data per image:\n')
            f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n')
                    
            for image_id_count, (image_path, pose) in tqdm(enumerate(zip(self.image_paths, self.poses)),total=len(self.image_paths)):

                print("image_path", image_path)
                cam_center = pose[:3, 3]
                # draw on matplotlib for cam center and pointcloud
                image_name = os.path.basename(image_path)
                # change pose to world to camera poses 
                pose_w2c = np.linalg.inv(pose)
                # inverse pose
                quaternion, translation = matrix_to_quaternion_and_translation(pose_w2c)
                # Write image data in COLMAP format
                image_id = image_id_count + 1
                camera_id = 1  # Assuming each image has a unique camera, change if needed
                qw, qx, qy, qz = quaternion  # Quaternion (w, x, y, z)
                tx, ty, tz = translation  # Translation (tx, ty, tz)
                f.write(f'{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n')

                # after image, we need to add keypoints
                # empty keypoint
                f.write('\n')
        # save empty points3d
        points_file_path = os.path.join(self.output_path, 'points3D.txt') 
        with open(points_file_path, 'w') as f:
            f.write('# 3D point list with one line of data per point:\n')
            f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        
        print("finish save_images_txt")


def matrix_to_quaternion_and_translation(matrix):
    """Convert 4x4 camera-to-world matrix to quaternion and translation."""
    rotation_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Convert rotation matrix to quaternion (x, y, z, w)
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    # change x y z w to w x y z
    quaternion = np.roll(quaternion, 1)

    return quaternion, translation

