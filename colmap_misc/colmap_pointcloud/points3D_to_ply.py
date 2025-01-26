from read_write_model import read_points3D_binary
import argparse
import numpy as np
import trimesh

def arg_parser():
    parser = argparse.ArgumentParser(description='Convert points3D to ply')
    parser.add_argument('--input', type=str, required=True, help='Input points3D file')
    parser.add_argument('--output', type=str, required=True, help='Output ply file')
    args = parser.parse_args()
    return args

def main():
    args = arg_parser()
    input_file = args.input
    output_file = args.output

    points3D = read_points3D_binary(input_file)
    print("length of points3D: ", len(points3D))
    vertices = []
    colors = []
    for point3D_id in points3D:
        point3D = points3D[point3D_id]
        vertices.append(point3D.xyz)
        colors.append(point3D.rgb)

    vertices = np.array(vertices)
    colors = np.array(colors)

    cloud = trimesh.PointCloud(vertices, colors=colors)
    cloud.export(output_file)


if __name__ == '__main__':
    main()