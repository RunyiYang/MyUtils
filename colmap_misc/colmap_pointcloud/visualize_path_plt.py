import json
import numpy as np
from camera_pose_visualizer import CameraPoseVisualizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import os
    org_path = '/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/iccv_2025/colmap_misc/colmap_pointcloud/debug_pose.npy' # 
    org_cam_poses_c2w = np.load(org_path, allow_pickle=True)

    print("x lim", np.min(org_cam_poses_c2w[:,0,-1]), np.max(org_cam_poses_c2w[:,0,-1]))
    print("y lim", np.min(org_cam_poses_c2w[:,1,-1]), np.max(org_cam_poses_c2w[:,1,-1]))
    print("z lim", np.min(org_cam_poses_c2w[:,2,-1]), np.max(org_cam_poses_c2w[:,2,-1]))

    x_lim_min = np.min(org_cam_poses_c2w[:,0,-1]) * 1.3
    x_lim_max = np.max(org_cam_poses_c2w[:,0,-1]) * 1.3
    y_lim_min = np.min(org_cam_poses_c2w[:,1,-1]) * 1.3
    y_lim_max = np.max(org_cam_poses_c2w[:,1,-1]) * 1.3
    z_lim_min = np.min(org_cam_poses_c2w[:,2,-1]) * 1.3
    z_lim_max = np.max(org_cam_poses_c2w[:,2,-1]) * 1.3


    # argument : the minimum/maximum value of x, y, z
    visualizer = CameraPoseVisualizer([x_lim_min, x_lim_max], [y_lim_min, y_lim_max], [z_lim_min, z_lim_max])

    K  = np.array([[1920.0, 0, 960, 0],[0, 1080.0, 540, 0], [0,0,1, 0],[0,0, 0,1]])
    img_size = np.array([1920,1080]) # W, H

    for index, frame_i in enumerate(org_cam_poses_c2w[::10]):
        visualizer.extrinsic2pyramid(frame_i, plt.cm.rainbow(index / len(org_cam_poses_c2w)), focal_len_scaled=0.2, aspect_ratio=img_size[0]/img_size[1])

    # visualizer.show()
    visualizer.save_fig()
    # import matplotlib.pyplot as plt
    # plt.plot(org_cam_poses_c2w[:,0,-1], org_cam_poses_c2w[:,1,-1])
    # plt.show()
