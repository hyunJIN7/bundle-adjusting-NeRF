import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
rgb,depth,intrinsic 필요 
"""

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # parser.add_argument("--half_res", action='store_true',
    #                     help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--basedir", type=str, default='./pointcloud',
                        help='input data directory')
    parser.add_argument("--datadir", type=str, default='',
                        help='input data directory')
    parser.add_argument("--expname", type=str, default='/fern_test_origin',
                        help='experiment name')

    return parser


# def draw_image(color,depth):
#     plt.subplot(1, 2, 1)
#     plt.title('grayscale image')
#     plt.imshow(color)
#     plt.subplot(1, 2, 2)
#     plt.title('depth image')
#     plt.imshow(depth)
#     plt.show()

def draw_image(rgbd):
    plt.subplot(1, 2, 1)
    plt.title('grayscale image')
    plt.imshow(rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title('depth image')
    plt.imshow(rgbd.depth)
    plt.show()

def load_data(args):

    color_dir = os.path.join(args.basedir, 'color')
    depth_dir = os.path.join(args.basedir, 'depth')

    print("Read dataset")
    color_list = os.listdir(color_dir)
    depth_list = os.listdir(depth_dir)

    all_color = []
    all_depth = []
    all_rgbd = []
    for i in range(len(color_list)):
        color_raw = o3d.io.read_image(os.path.join(color_dir, color_list[i]))
        depth_raw = o3d.io.read_image(os.path.join(depth_dir, depth_list[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1.0)

        all_color.append(color_raw)
        all_depth.append(depth_raw)
        all_rgbd.append(rgbd_image)
        print(rgbd_image)
    draw_image(all_rgbd[0])
    return all_color, all_depth, all_rgbd

def draw_pcd(args):

    all_color,all_depth,all_rgbd = load_data(args)
    all_pcd= o3d.geometry.PointCloud()

    raw_intr = [1510.687378, 1510.687378, 963.860107, 722.338562]
    ori_size = (1920, 1440)
    size = (640, 480)
    # intr[0, :] /= (ori_size[0] / size[0])
    # intr[1, :] /= (ori_size[1] / size[1])
    raw_intr[0] /= (ori_size[0] / size[0])
    raw_intr[2] /= (ori_size[0] / size[0])
    raw_intr[1] /= (ori_size[1] / size[1])
    raw_intr[3] /= (ori_size[1] / size[1])
    intr = [[raw_intr[0], 0, raw_intr[2]],
            [0, raw_intr[1], raw_intr[3]],
            [0, 0, 1]]


    # for rgbd_image in enumerate(all_rgbd):
    for i in range(len(all_rgbd)):
        # 나중에 for안으로 넣어라
        cam_intr = o3d.camera.PinholeCameraIntrinsic(size[0], size[1], raw_intr[0], raw_intr[1], raw_intr[2],
                                                     raw_intr[3])
        cam_intr.intrinsic_matrix = intr
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = cam_intr
        # cam.extrinsic = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 1.]])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            all_rgbd[i],
            cam.intrinsic,
            # cam.extrinsic
        )

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])
        all_pcd += pcd
    all_pcd_down = all_pcd.voxel_down_sample(voxel_size=0.0005)
    #all_pcd_down = all_pcd

    o3d.visualization.draw_geometries([all_pcd_down])#, zoom=0.35) zoom error


#python pointcloud/create_rgbdepth2pointcloud.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    draw_pcd(args)
