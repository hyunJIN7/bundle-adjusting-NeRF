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

# DEPTH_WIDTH = 256
# DEPTH_HEIGHT = 192
RGB_WIDTH = 1920
RGB_HEIGHT = 1440
MAX_DEPTH = 20.0
np.random.seed(0)

"""
conda activate StrayVisualizer-main
python data/process_strayscanner_data.py --num_train=3 --basedir ./data/strayscanner/lab_computer_3 --depth_bound2=0.2 --depth_bound1=0.7
python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/lab_computer_5 --depth_bound2=0.2 --depth_bound1=0.7
python data/process_strayscanner_data.py --num_train=7 --basedir ./data/strayscanner/trashcan01_7 --depth_bound2=0.2 --depth_bound1=0.7

python data/process_strayscanner_data.py --num_train=10  --basedir ./data/strayscanner/a1_5 --depth_bound2=0.2 --depth_bound1=0.7

conda activate StrayVisualizer-main
python data/process_strayscanner_data.py --num_train=50  --basedir ./data/strayscanner/f_box --depth_bound2=0.2 --depth_bound1=0.7
python data/process_strayscanner_data.py --num_train=10  --basedir ./data/strayscanner/tree --depth_bound2=0.2 --depth_bound1=0.7


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
    rgb_path = "{}/rgb_{}".format(args.basedir, split)
    make_dir(rgb_path)

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
    train_test_index = np.hstack((train_index,test_index))
    train_test_index.sort()


    rgbs = np.array(data['rgb'])
    poses = np.array(data['odometry'])
    pose_fname = "{}/odometry_{}.csv".format(args.basedir, split)
    pose_file = open(pose_fname, 'w')  # ,newline=','
    wr = csv.writer(pose_file)
    for i, (rgb,pose,) in enumerate(zip(rgbs,poses)):
        # pose :  timestamp, frame, x, y, z, qx, qy, qz, qw
        skvideo.io.vwrite(os.path.join(rgb_path, str(int(pose[1])).zfill(5) + '.png'), rgb)
        wr.writerow(pose)
    pose_file.close()


def precompute_depth_sampling(origin_near, origin_far, depth, confidence):
    # TODO : 지금 기준은 confidence , 성능 구리면 depth 값 기준으로도 더 조건 추가 4.5 이상이면 해보고 별로면
    depth_min, depth_max = origin_near, origin_far
    # [N,H,W]
    depth = torch.tensor(depth)
    confidence = torch.tensor(confidence)

    depth = depth[..., None]  # [N,H,W,1]
    confidence = confidence[..., None]  # [N,H,W,1]
    near = torch.ones_like(depth)  # [N,H,W,1]
    far = torch.ones_like(depth)

    condi2 = confidence[..., 0] == 2  # [N,H,W]
    bound2 = args.depth_bound2
    near[condi2] = torch.clamp(depth[condi2] - bound2, min=0)
    far[condi2] = depth[condi2] + bound2

    condi1 = confidence[..., 0] == 1
    bound1 = args.depth_bound1
    near[condi1] = torch.clamp(depth[condi1] - bound1, min=0)
    far[condi1] = depth[condi1] + bound1

    condi0 = confidence[..., 0] == 0
    if args.use_confi0_depth > 0:
        # consider depth
        near[condi0] = 2  # torch.clamp(depth[condi0]-0.3,max=4)
        far[condi0] = torch.clamp(depth[condi0] + 0.3, min=depth_max)
    else:
        # near = 4
        near[condi0] = 4  # torch.clamp(depth[condi0]+0.3,min=4)
        far[condi0] = torch.clamp(depth[condi0] + 0.3, min=depth_max)

    test_near, test_far = near[..., 0], far[..., 0]
    test_depth = depth[..., 0]
    return near[..., 0], far[..., 0]  # [B,H*W]


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

    process_stray_scanner(args, data)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
