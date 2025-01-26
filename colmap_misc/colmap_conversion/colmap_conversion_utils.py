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
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"



def depths_exr_to_inv_depths_png(depths_exr_path, inv_depths_png_path):
    depth_exr, _ = load_depth(depths_exr_path)
    inv_depth = (1.0 * 2**16) / depth_exr
    # change to float 32
    inv_depth = inv_depth.astype(np.float32)
    # times anther 2**16 then save
    # save to png file
    cv2.imwrite(inv_depths_png_path, inv_depth)




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


class ImageDepth2Colmap():
    def __init__(self, image_paths, depth_paths, poses, pointclouds_path, intrinsics, output_path):
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.image_paths_out = []
        self.depth_paths_out = []
        self.pointclouds_path = pointclouds_path
        self.intrinsics = intrinsics
        self.output_path = output_path
        self.poses = poses

    # def collect_files_light(self):
    #     os.makedirs(self.output_path, exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'images'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'depths_exr'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'depths'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'sparse', 'known'), exist_ok=True)
    #     os.makedirs(os.path.join(self.output_path, 'sparse', '0'), exist_ok=True)
    #     new_transforms_path = os.path.join(self.output_path, 'sparse', 'known' , 'transforms.json')
    #     new_transforms = {}
    #     new_transforms['train'] = {}
    #     new_transforms['test'] = {}
    #     for i, (img_path, depth_path) in tqdm(enumerate(zip(self.image_paths, self.depth_paths)),total=len(self.image_paths) ): 

    def collect_files(self, overwrite=False):
        # copy images and depth to output_path/rectified/images and output_path/rectified/depth
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'depths_exr'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'depths'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'sparse', 'known'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'sparse', '0'), exist_ok=True)

        # also save a new transform.json
        copy_map = {}
        new_transforms_path = os.path.join(self.output_path, 'sparse', 'known' , 'transforms.json')
        new_transforms = {}
        new_transforms['train'] = {}
        new_transforms['test'] = {}
        new_transforms['train_dense'] = {}

        for i, (img_path, depth_path) in tqdm(enumerate(zip(self.image_paths, self.depth_paths)),total=len(self.image_paths) ):
            img_target_path = os.path.join(self.output_path, 'images', f"{str(i).zfill(8)}.png")
            depth_target_path = os.path.join(self.output_path, 'depths_exr', f"{str(i).zfill(8)}.exr")
            inv_depth_target_path = os.path.join(self.output_path, 'depths', f"{str(i).zfill(8)}.png")
   
            if not overwrite:
                if not os.path.exists(img_target_path):
                    shutil.copy(img_path, img_target_path)
                if not os.path.exists(depth_target_path):
                    shutil.copy(depth_path, depth_target_path)
            else:
                shutil.copy(img_path, img_target_path)
                shutil.copy(depth_path, depth_target_path)
            self.image_paths_out.append(img_target_path)
            self.depth_paths_out.append(depth_target_path)
            split = img_path.split("/")[-3]
            block_name = img_path.split("/")[-2]
            data_dict = {str(i).zfill(8): self.poses[i].tolist()}
            if block_name not in new_transforms[split]:
                new_transforms[split][block_name] = [data_dict]
            else:
                new_transforms[split][block_name].append(data_dict)
            # new_transforms[split].append(data_dict)
            if not os.path.exists(inv_depth_target_path):
                depths_exr_to_inv_depths_png(depth_target_path, inv_depth_target_path)

        with open(new_transforms_path, 'w') as f:
            json.dump(new_transforms, f, indent=4)


    def save_cameras_txt(self):
        """
        Save camera intrinsics to cameras.txt in COLMAP format.
        
        Args:
        - cameras (list): List of dictionaries with camera model parameters.
        - output_path (str): Path to save the cameras.txt file.
        """
        camera_file_path = os.path.join(self.output_path, 'sparse', 'known' , 'cameras.txt')
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


    def init_globl_points3d(self):
        # load pointclouds_path .ply file
        self.points3d = {}
        print("load pointclouds_path", self.pointclouds_path)
        points3d = trimesh.load_mesh(self.pointclouds_path)
        print("load points number", len(points3d.vertices))
        color3d = points3d.visual.vertex_colors
        print("===== color3d ======", color3d.shape)
        self.color3d = color3d[:, :3]  # only RGB
        self.points3d['POINT3D_ID'] = np.arange(len(points3d.vertices)) + 1
        self.points3d['XYZ'] = points3d.vertices * 100  # cm to meter
        self.points3d['TRACK'] = [[] for _ in range(len(points3d.vertices))]


 
    def save_images_txt(self, depth_threshold=0.2, depth_max=500, debug=False, tmp_save=False, show_t=False, start_idx = 0, end_idx=None):
        # if not tmp_save:
        assert start_idx < len(self.image_paths_out), f"start_idx {start_idx} should be smaller than len(self.image_paths_out) {len(self.image_paths_out)}"
        if end_idx is None or end_idx > len(self.image_paths_out):
            end_idx = len(self.image_paths_out)

        image_file_path = os.path.join(self.output_path,'sparse', 'known' , 'images.txt')
        if tmp_save:
            tmp_save_path = os.path.join(self.output_path, 'sparse', 'known', 'tmp')
            os.makedirs(tmp_save_path, exist_ok=True)

        assert len(self.image_paths_out) == len(self.depth_paths_out) == len(self.poses), f"len(self.image_paths_out) {len(self.image_paths_out)} != len(self.depth_paths_out) {len(self.depth_paths_out)} != len(self.poses) {len(self.poses)}"

        unique_img_id_list = []
        # if image_file_path exist, remove it
        if os.path.exists(image_file_path):
            os.remove(image_file_path)

        image_file_path = os.path.join(self.output_path,'sparse', 'known' , f'images_{str(start_idx).zfill(8)}_{str(end_idx).zfill(8)}.txt')

        with open(image_file_path, 'w') as f:
            f.write('# Image list with one line of data per image:\n')
            f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n')
                    
            for image_id_count, (image_path, depth_path, pose) in tqdm(enumerate(zip(self.image_paths_out, self.depth_paths_out,  self.poses)),total=len(self.image_paths_out)):
                
                if image_id_count < start_idx or image_id_count >= end_idx:
                    continue
                print("image_path", image_path)
                print("depth_path", depth_path)
                time_0 = time.time()
                cam_center = pose[:3, 3]
                # draw on matplotlib for cam center and pointcloud
                image_name = os.path.basename(image_path)
                # change pose to world to camera poses 
                pose_w2c = np.linalg.inv(pose)
                # inverse pose
                quaternion, translation = matrix_to_quaternion_and_translation(pose_w2c)
                # Write image data in COLMAP format
                image_id = image_id_count + 1
                assert image_id not in unique_img_id_list, f"image_id {image_id} already exists"
                camera_id = 1  # Assuming each image has a unique camera, change if needed
                qw, qx, qy, qz = quaternion  # Quaternion (w, x, y, z)
                tx, ty, tz = translation  # Translation (tx, ty, tz)
                f.write(f'{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n')
                unique_img_id_list.append(image_id)

                # after image, we need to add keypoints
                render_depth_map, mask_map, used_points3d_id = render_3d_world_to_camera_opencv(self.points3d, pose_w2c, self.intrinsics, depth_max=depth_max)
                time_1 = time.time()
                if debug:
                    render_depth_map_save = render_depth_map.copy()
                    render_depth_map_save = np.clip(render_depth_map_save, 0, 255)
                    render_depth_map_save = render_depth_map_save.astype(np.uint8)
                    cv2.imwrite('./aerial_debug/aerial_render_depth_map_abs_check.png', render_depth_map_save)

                    # save depth maask, mask map and id 
                    # write depth map
                    # depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    # cv2.imwrite('./depth_debug.png', depth_map_norm)
                    # write mask map
                    # mask_map = mask_map * 255
                    # cv2.imwrite('./mask_debug.png', mask_map)
                    # write img
                    shutil.copy(image_path, './aerial_debug/aerial_img_debug.png')
                # depth_map  = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            
                # # write depth_map
                # depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                # cv2.imwrite('./depth_map.png', depth_map_norm)
                depth_map, invalid_mask = load_depth(depth_path)
                if debug:
                    print("depth_map range", depth_map.shape, depth_map.min(), depth_map.max(), depth_map.mean())
                    depth_map_save = depth_map.copy()
                    depth_map_save =  np.clip(depth_map_save, 0, 255)
                    depth_map_save = depth_map_save.astype(np.uint8)

                    cv2.imwrite('./aerial_debug/aerial_depth_map_abs_check.png', depth_map_save)


                masked_depth = np.where(mask_map == 1, depth_map, 0)
                
                # we calculate the difference of render_depth_map and masked_depth
                depth_diff = np.abs(render_depth_map - masked_depth)
                # get indices where depth_diff is smaller than 0.1 # 10 cm
                mask_depth_loss = depth_diff <depth_threshold
                # combine mask_depth_loss and mask_map
                mask_keypoint = mask_map * mask_depth_loss
                print("matched points", np.sum(mask_keypoint))
                time_2 = time.time()
                if debug:
                    mask_keypoint_save = mask_keypoint.astype(np.uint8)*255
                    # mask_keypoint.clip(0, 255).astype(np.uint8)
                    cv2.imwrite('./aerial_debug/aerial_matched_points_abs_check.png', mask_keypoint_save)

                # get the points3d_id
                # get the points3d_xyz
                # update 2D images and its keypoints,  (X, Y, POINT3D_ID)
                # write keypoints
                keypoints_indices = np.where(mask_keypoint == 1) # H, W 
                keypoints = np.zeros((len(keypoints_indices[0]), 3), dtype=np.int64)
                keypoints[:, 0] = keypoints_indices[1]
                keypoints[:, 1] = keypoints_indices[0]
                keypoints[:, 2] = used_points3d_id[keypoints_indices[0], keypoints_indices[1]]
            
                # update points3d TRACK[] as (IMAGE_ID, POINT2D_IDX)
                if not tmp_save:
                    for count, (u, v, point3d_id) in enumerate(keypoints):
                        self.points3d['TRACK'][point3d_id-1].append([image_id, count]) # this might be too long, better to make points3d also writable
                else:
                    # for every image, we save a self.points3d['track']
                    points3d_track_file = os.path.join(tmp_save_path, f"{str(image_id).zfill(8)}_track_length.json")
                    track_length = {}
                    for count, (u, v, point3d_id) in enumerate(keypoints):
                        if point3d_id not in track_length:
                            track_length[int(point3d_id)] = [[int(image_id), int(count)]]
                        else:
                            track_length[int(point3d_id)].append([int(image_id), int(count)])
                    with open(points3d_track_file, 'w') as fp:
                        json.dump(track_length, fp)
                    # no with open, directly dump to points3d_track_file
                    json.dump(track_length, open(points3d_track_file, 'w'))
                time_3 = time.time()
                #     else:
                #         # tmp save track length and load  
                #         track_length_file = os.path.join(tmp_save_path, f"{str(point3d_id).zfill(8)}_track_length.npy")
                #         if os.path.exists(track_length_file):
                #             track_length = np.load(track_length_file)
                #             track_length = track_length.tolist()
                #         else:
                #             track_length = []
                #         track_length.append([image_id, count])
                #         track_length = np.array(track_length)
                #         np.save(track_length_file, track_length)
                # time_3 = time.time()

                if debug and image_id_count==0:
                    raise ValueError("stop here")

                if show_t:
                    print("render time", time_1 - time_0)
                    print("depth time", time_2 - time_1)
                    print("keypoint time", time_3 - time_2)
                # write keypoints
                # save key points as (X, Y, POINT3D_ID)
                keypoints = keypoints.tolist()
                # if key point is empty, we add empty line
                if len(keypoints) == 0:
                    f.write('\n')
                    continue
                for count, keypoint in enumerate(keypoints):
                    if count == len(keypoints) - 1:
                        f.write(f'{keypoint[0]} {keypoint[1]} {keypoint[2]}\n')
                    else:
                        f.write(f'{keypoint[0]} {keypoint[1]} {keypoint[2]} ')
                        # print("write keypoint", keypoint)


        # save points3d
        points3d_file_path = os.path.join(self.output_path,'sparse', 'known' , f'points3D_{str(start_idx).zfill(8)}_{str(end_idx).zfill(8)}.txt')
        with open(points3d_file_path, 'w') as f:
            f.write('# 3D point list with one line of data per point:\n')
            f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
            for i, point3d_id in enumerate(self.points3d['POINT3D_ID']):
                # only save points if track is not empty
                point3d = self.points3d['XYZ'][i]
                r, g, b = self.color3d[i]
                error = 0
                if not tmp_save:
                    track = self.points3d['TRACK'][i]
                else:
                    # get all track length in tmp_save_path
                    track_length_files = os.listdir(tmp_save_path)
                    track_length_files = [track_length_file for track_length_file in track_length_files if track_length_file.endswith('.json')]
                    track = []
                    for track_length_file in track_length_files:
                        with open(os.path.join(tmp_save_path, track_length_file), 'r') as f:
                            track_length = json.load(f)
                        if str(point3d_id) in track_length:
                            track += track_length[str(point3d_id)]

                if len(track) > 0:
                    track_str = ' '.join([f'{t[0]} {t[1]}' for t in track])
                    f.write(f'{point3d_id} {point3d[0]} {point3d[1]} {point3d[2]} {r} {g} {b} {error} {track_str}\n')


    def merge_images_points_txt(self):
        all_files = os.listdir(os.path.join(self.output_path,'sparse', 'known'))
        images_file = [file for file in all_files if file.startswith('images_')]
        points3d_file = [file for file in all_files if file.startswith('points3D_')]
        # merge images
        images_file = sorted(images_file)
        new_image_file = os.path.join(self.output_path,'sparse', 'known', 'images.txt')
        with open (new_image_file, 'w') as f_new:
            f_new.write('# Image list with one line of data per image:\n')
            f_new.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n')
            for i, file in enumerate(images_file):
                with open(os.path.join(self.output_path,'sparse', 'known', file), 'r') as f_read:
                    for line in f_read:
                        if line.startswith('#'):
                            continue
                        f_new.write(line)

        # merge points3d
        new_points3d_file = os.path.join(self.output_path,'sparse', 'known', 'points3D.txt')

        points3d_file = sorted(points3d_file)
        track_summary =  [[] for _ in range(len(self.points3d['XYZ']))]
        for i, file in enumerate(points3d_file):
            with open(os.path.join(self.output_path,'sparse', 'known', file), 'r') as f_read:
                for line in f_read:
                    if line.startswith('#'):
                        continue
                    line = line.strip().split()
                    point3d_id = int(line[0])
                    track = line[8:]
                    # print("track", track)
                    track = [[int(track[i]), int(track[i+1])] for i in range(0, len(track), 2)]
                    track_summary[point3d_id-1] += track

        with open(new_points3d_file, 'w') as f:
            f.write('# 3D point list with one line of data per point:\n')
            f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
            for i, point3d_id in enumerate(self.points3d['POINT3D_ID']):
                # only save points if track is not empty
                point3d = self.points3d['XYZ'][i]
                r, g, b = self.color3d[i]
                error = 0
                track =  track_summary[i]
                if len(track) > 0:
                    track_str = ' '.join([f'{t[0]} {t[1]}' for t in track])
                    f.write(f'{point3d_id} {point3d[0]} {point3d[1]} {point3d[2]} {r} {g} {b} {error} {track_str}\n')


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

