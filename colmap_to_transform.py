#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# This script converts COLMAP-exported camera data into a transforms.json file
# compatible with NeRF (Neural Radiance Fields). It supports both text and binary
# COLMAP outputs (`cameras.txt` / `cameras.bin` and `images.txt` / `images.bin`).

import argparse
import json
import math
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import struct

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP export to NeRF format transforms.json.")
    
    parser.add_argument("--images", default="images", help="Path to the folder containing images.")
    parser.add_argument("--text", default="colmap_text", help="Path to the folder with COLMAP text output.")
    parser.add_argument("--binary", action="store_true", help="Use binary COLMAP files (cameras.bin and images.bin) instead of text.")
    parser.add_argument("--aabb_scale", default=32, type=int, choices=[1, 2, 4, 8, 16, 32, 64, 128], 
                        help="Scene scale factor. 1=scene fits in unit cube; power of 2 up to 128.")
    parser.add_argument("--out", default="transforms.json", help="Path to the output transforms.json file.")
    parser.add_argument("--skip_early", default=0, type=int, help="Number of initial images to skip.")
    parser.add_argument("--keep_colmap_coords", action="store_true", 
                        help="Keep original COLMAP coordinate frame.")
    
    return parser.parse_args()

def read_next_bytes(fid, num_bytes, format_char_sequence, is_string=False):
    data = fid.read(num_bytes)
    if is_string:
        return data.decode('utf-8')
    return struct.unpack(format_char_sequence, data)

def read_cameras_binary(file_path):
    cameras = {}
    with open(file_path, "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(f, 8, "I")[0]
            model_id = read_next_bytes(f, 2, "H")[0]
            width = read_next_bytes(f, 4, "I")[0]
            height = read_next_bytes(f, 4, "I")[0]
            params = read_next_bytes(f, 8 * 4, "dddd")

            cameras[camera_id] = {
                "w": width,
                "h": height,
                "fl_x": params[0],
                "fl_y": params[1],
                "cx": params[2],
                "cy": params[3]
            }
    return cameras

def read_images_binary(file_path):
    images = {}
    with open(file_path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(f, 8, "I")[0]
            qvec = read_next_bytes(f, 8 * 4, "dddd")
            tvec = read_next_bytes(f, 8 * 3, "ddd")
            camera_id = read_next_bytes(f, 8, "I")[0]
            name_length = read_next_bytes(f, 4, "I")[0]
            name = read_next_bytes(f, name_length, f"{name_length}s", is_string=True)

            images[image_id] = {
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id,
                "name": name
            }
    return images

def qvec2rotmat(qvec):
    """Convert a quaternion to a rotation matrix."""
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
        ]
    ])

def rotmat(a, b):
    """Compute the rotation matrix aligning vector a to vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))

if __name__ == "__main__":
    args = parse_args()

    AABB_SCALE = args.aabb_scale
    SKIP_EARLY = args.skip_early
    IMAGE_FOLDER = args.images
    TEXT_FOLDER = args.text
    OUT_PATH = args.out

    if args.binary:
        cameras = read_cameras_binary(os.path.join(TEXT_FOLDER, "cameras.bin"))
        images = read_images_binary(os.path.join(TEXT_FOLDER, "images.bin"))
    else:
        cameras = {}
        with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                els = line.split()
                camera_id = int(els[0])
                cameras[camera_id] = {
                    "w": float(els[2]),
                    "h": float(els[3]),
                    "fl_x": float(els[4]),
                    "fl_y": float(els[4]),
                    "cx": float(els[5]),
                    "cy": float(els[6])
                }

        images = {}
        with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(list(map(float, elems[1:5])))
                tvec = np.array(list(map(float, elems[5:8])))
                camera_id = int(elems[8])
                name = os.path.join(IMAGE_FOLDER, '_'.join(elems[9:]))

                images[image_id] = {
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name
                }

    if not cameras:
        print("No cameras found!")
        exit(1)

    out = {
        "frames": [],
        "aabb_scale": AABB_SCALE
    }

    up = np.zeros(3)
    for image_id, image_data in tqdm(images.items()):
        qvec = image_data["qvec"]
        tvec = image_data["tvec"]
        camera = cameras[image_data["camera_id"]]
        
        R = qvec2rotmat(-qvec)
        t = tvec.reshape([3, 1])
        c2w = np.linalg.inv(np.vstack([np.hstack([R, t]), [0, 0, 0, 1]]))

        if not args.keep_colmap_coords:
            c2w[0:3, 2] *= -1
            c2w[0:3, 1] *= -1
            c2w = c2w[[1, 0, 2, 3], :]
            c2w[2, :] *= -1
            up += c2w[0:3, 1]

        out["frames"].append({
            "file_path": image_data["name"],
            "transform_matrix": c2w.tolist()
        })

    if not args.keep_colmap_coords:
        up = up / np.linalg.norm(up)
        R = rotmat(up, [0, 0, 1])
        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, np.array(f["transform_matrix"]))

    with open(OUT_PATH, "w") as outfile:
        json.dump(out, outfile, indent=2)

    print(f"Transforms saved to {OUT_PATH}")
