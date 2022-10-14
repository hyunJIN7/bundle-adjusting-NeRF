import csv
# import pickle
# import torch
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


DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

#  conda activate StrayVisualizer-main
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/computer
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/dinosaur
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/xyz
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/xyz2 --num_train=200
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/MPIL01 --num_train=200
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--num_train", type=int, default=130,
                        help='number of train data')
    parser.add_argument("--num_test", type=int, default=20,
                        help='number of train data')
    return parser

# def load_depth(path, confidence=None):
#     depth_mm = np.array(Image.open(path))
#     depth_m = depth_mm.astype(np.float32) / 1000.0
#     return depth_m

def load_depth(path, confidence=None):
    extension = os.path.splitext(path)[1]
    if extension == '.npy':
        depth_mm = np.load(path)
    elif extension == '.png':
        depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


def load_confidence(path):
    return np.array(Image.open(path))

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def process_stray_scanner(args, data,split='train'):
    rgb_path = "{}/rgb_{}".format(args.basedir, split)
    depth_path = "{}/depth_{}".format(args.basedir, split)
    confidence_path = "{}/confidence_{}".format(args.basedir, split)
    make_dir(rgb_path)
    make_dir(depth_path)
    make_dir(confidence_path)

    n = data['odometry'].shape[0]
    num_train = args.num_train
    num_val = 4
    num_test = args.num_test


    all_index = np.arange(n)
    train_val_index = np.linspace(0, n, num_train+num_val, endpoint=False, dtype=int)
    train_index = train_val_index[:-num_val]
    val_index = train_val_index[-num_val:]
    test_index = np.delete(all_index,train_val_index)
    test_index = np.random.choice(test_index,num_test,replace=False)
    # print("train ", train_index)
    # print("val ", val_index)
    # print("test_index ", test_index)
    # print("train_val_index ", train_val_index)


    # print("list rgb ",data['rgb'])
    rgbs = np.array(data['rgb'])
    # print("np rgb ", rgbs)
    depths = np.array(data['depth'])
    confidences = np.array(data['confidence'])
    poses = np.array(data['odometry'])
    if split == 'train':
        rgbs = rgbs[train_index]
        depths = depths[train_index]
        confidences = confidences[train_index]
        poses = poses[train_index]
    elif split == 'val':
        rgbs = rgbs[val_index]
        depths = depths[val_index]
        confidences = confidences[val_index]
        poses = poses[val_index]
    elif split == 'test':
        rgbs = rgbs[test_index]
        depths = depths[test_index]
        confidences = confidences[test_index]
        poses = poses[test_index]
    else:
        rgbs = rgbs[train_val_index]
        depths = depths[train_val_index]
        confidences = confidences[train_val_index]
        poses = poses[train_val_index]


    pose_fname = "{}/odometry_{}.csv".format(args.basedir, split)
    pose_file = open(pose_fname,'w')#,newline=','
    wr = csv.writer(pose_file)
    for i, (rgb, depth, confidence, pose) in enumerate(zip(rgbs, depths,confidences,poses)):
        #pose :  timestamp, frame, x, y, z, qx, qy, qz, qw
        cv2.imwrite(os.path.join(rgb_path, str(int(pose[1])).zfill(5) + '.png'), rgb)
        np.save(os.path.join(depth_path, str(int(pose[1])).zfill(5) + '.npy'), depth)
        np.save(os.path.join(confidence_path, str(int(pose[1])).zfill(5) + '.npy'), confidence)
        # cv2.imwrite(os.path.join(depth_path, str(int(pose[1])).zfill(5) + '.npy'), depth)
        # cv2.imwrite(os.path.join(confidence_path, str(int(pose[1])).zfill(5) + '.npy'), confidence)
        wr.writerow(pose)
    pose_file.close()


def main(args):
    # data load
    data = {}
    # intrinsics = np.loadtxt(os.path.join(args.basedir, 'camera_matrix.csv'), delimiter=',')
    # data['intrinsics'] = intrinsics
    odometry = np.loadtxt(os.path.join(args.basedir, 'odometry.csv'), delimiter=',', skiprows=1)
    data['odometry'] = odometry

    depth_dir = os.path.join(args.basedir, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]
    data['depth_frames'] = depth_frames

    rgbs=[]
    confidences = []
    depths = []
    video_path = os.path.join(args.basedir, 'rgb.mp4')
    video = skvideo.io.vreader(video_path)
    for i, (T_WC, rgb) in enumerate(zip(data['odometry'], video)):
        #load confidence
        confidence = load_confidence(os.path.join(args.basedir, 'confidence', f'{i:06}.png'))
        confidences.append(confidence)

        #load depth
        print(f"Integrating frame {i:06}", end='\r')
        depth_path = data['depth_frames'][i]
        depth = load_depth(depth_path)
        depths.append(depth)

        #rgb image
        rgb = Image.fromarray(rgb)
        rgb = rgb.resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb = np.array(rgb)
        rgbs.append(rgb)
        # cv2.imwrite(os.path.join(rgb_img_path, str(i).zfill(5) + '.jpg'), rgb)

    data['confidence'] = confidences
    data['depth']=depths
    data['rgb']=rgbs

    split = ['train', 'val', 'test','train_val']
    for mode in split:
        process_stray_scanner(args,data,mode)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
