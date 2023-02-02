import csv
# import pickle
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import os
import numpy as np
# import imageio
# import json
# from transforms3d.quaternions import quat2mat
# from skimage import img_as_ubyte
from PIL import Image
import skvideo.io
import cv2
import random
from scipy.spatial.transform import Rotation

# DEPTH_WIDTH = 256
# DEPTH_HEIGHT = 192
RGB_WIDTH = 1920
RGB_HEIGHT = 1440
MAX_DEPTH = 20.0
np.random.seed(0)

"""
conda activate StrayVisualizer-main
conda activate StrayVisualizer-main
python data/process_strayscanner_data_for_dsnerf.py --num_train=10  --basedir ./data/strayscanner/test1_ds


"""

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--num_train", type=int, default=120,
                        help='number of train data')
    parser.add_argument("--num_test", type=int, default=15,
                        help='number of train data')

    parser.add_argument("--near_range", type=int, default=2,
                        help='near near range')
    parser.add_argument("--far_range", type=int, default=6,
                        help='far near range')

    parser.add_argument("--use_confi0_depth", type=int, default=1,
                        help='far near range')
    parser.add_argument("--depth_bound2", type=float, default=0.4,
                        help='condi2 depth range')
    parser.add_argument("--depth_bound1", type=float, default=1,
                        help='condi1 depth range')
    return parser

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def process_stray_scanner(args, data, split='train'):
    basedir = "{}/{}".format(args.basedir,split)
    rgb_path = "{}/images".format(basedir)
    sparse_path = "{}/sparse".format(basedir)

    make_dir(basedir)
    make_dir(rgb_path)
    make_dir(sparse_path)

    # #cameras.txt
    H, W = data['rgb'][0].shape[:-1]
    intrinsic = data['intrinsics']
    camera_file = open(os.path.join(sparse_path, 'cameras.txt'), 'w')
    # camera_id, model, W, H , fx, fy , cx ,cy
    line = "1 PINHOLE " + str(W) + ' ' + str(H) + ' ' + str(intrinsic[0][0]) + ' ' + str(intrinsic[1][1]) + ' ' + str(
        intrinsic[0][2]) + ' ' + str(intrinsic[1][2])
    camera_file.write(line)
    camera_file.close()

    #points3D.txt
    points3D_file = open(os.path.join(sparse_path, 'points3D.txt'),'w')
    points3D_file.close()

    n = data['odometry'].shape[0]
    num_train = args.num_train
    num_val = 4
    num_test = args.num_test

    all_index = np.arange(n)
    train_val_index = np.linspace(0, n, num_train + num_val, endpoint=False, dtype=int)
    train_index = train_val_index[:-num_val]
    val_index = train_val_index[-num_val:]
    test_index = np.delete(all_index, train_val_index)
    test_index= test_index[np.linspace(0, test_index.shape[0], num_test, endpoint=False, dtype=int)]
    # train_test_index = np.hstack((train_index,test_index))
    # train_test_index.sort()

    rgbs = np.array(data['rgb'])
    poses = np.array(data['odometry'])
    main_index = train_index if split == 'train'else test_index
    rgbs = rgbs[main_index]
    poses = poses[main_index]


    pose_file = "{}/images.txt".format(sparse_path)
    lines = []
    for i, (rgb,pose) in enumerate(zip(rgbs,poses)):
        skvideo.io.vwrite(os.path.join(rgb_path, str(int(pose[1])).zfill(5) + '.png'), rgb)

        # pose :  timestamp, frame, x, y, z, qx, qy, qz, qw
        skvideo.io.vwrite(os.path.join(rgb_path, str(int(i)).zfill(5) + '.png'), rgb)
        # pose :  timestamp, frame, x, y, z, qx, qy, qz, qw
        # image_id(1,2,3,...),  qw, qx, qy, qz ,tx, ty, tz , camera_id, name(file name)

        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(pose[5:]).as_matrix()
        T_WC[:3, 3] = pose[2:5]
        c2w = np.linalg.inv(T_WC)
        quat = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        w = quat[-1]
        quat[1:] = quat[:3]
        quat[0] = w  # qw, qx, qy, qz

        t = np.reshape(c2w[:3, 3], (1,3)) # tx, ty ,tz

        line = []
        line.append(str(i+1))  #이미지 번호 바꿀까????? 원본 번호로 해도 되지 않을까

        new_pose = np.zeros(7)
        new_pose[:4] = quat
        new_pose[4:] = t
        for j in new_pose : line.append(str(j))
        line.append(str(1)) # camera_id
        line.append( str(int(pose[1])).zfill(5) + '.png') # name
        lines.append(' '.join(line) + '\n' + '\n')

    with open(pose_file, 'w') as f:
        f.writelines(lines)

def main(args):
    # data load
    data = {}
    intrinsics = np.loadtxt(os.path.join(args.basedir, 'camera_matrix.csv'), delimiter=',')
    data['intrinsics'] = intrinsics
    odometry = np.loadtxt(os.path.join(args.basedir, 'odometry.csv'), delimiter=',', skiprows=1)
    data['odometry'] = odometry

    depth_dir = os.path.join(args.basedir, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]
    data['depth_frames'] = depth_frames

    rgbs = []
    confidences = []
    depths = []
    video_path = os.path.join(args.basedir, 'rgb.mp4')
    video = skvideo.io.vreader(video_path)
    rgb_img_path = os.path.join(args.basedir, 'rgb')
    # make_dir(rgb_img_path)
    for i, (T_WC, rgb) in enumerate(zip(data['odometry'], video)):
        # rgb image
        rgb = np.array(Image.fromarray(rgb))
        rgbs.append(rgb)
    data['rgb'] = rgbs

    split = ['train', 'test']
    for mode in split:
        process_stray_scanner(args,data,mode)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
