from pathlib import Path
import os 
import sys

root_dir = '/srv/beegfs-benderdata/scratch/qimaqi_data/data/iccv_2025/datasets/Dataset/scripts/DL3DV-10K/1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f'
rgb_dir = root_dir / 'images_4'
colmap_dir = root_dir / 'colmap'
colmap_db_path = colmap_dir / 'database.db'
colmap_out_path = colmap_dir / 'sparse'
colmap_out_path.mkdir(exist_ok=True, parents=True)




sys.
!colmap feature_extractor \
--SiftExtraction.use_gpu 0 \
--SiftExtraction.upright {int(assume_upright_cameras)} \
--ImageReader.camera_model OPENCV \
--ImageReader.single_camera {int(share_intrinsics)} \
--database_path "{str(colmap_db_path)}" \
--image_path "{str(colmap_rgb_dir)}"
