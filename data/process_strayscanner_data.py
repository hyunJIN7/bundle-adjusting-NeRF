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

#conda activate StrayVisualizer-main
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/computer2
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/dinosaur
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--num_train", type=int, default=100,
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

def process_stray_scanner(args, data, split = 'train'):
    rgb_path = "{}/rgb_{}".format(args.basedir, split)
    depth_path = "{}/depth_{}".format(args.basedir, split)
    confidence_path = "{}/confidence_{}".format(args.basedir, split)
    make_dir(rgb_path)
    make_dir(depth_path)
    make_dir(confidence_path)

    num_train = args.num_train
    num_val = 4
    if split == 'train':
        rgbs = data['rgb'][:num_train]
        depths = data['depth'][:num_train]
        confidences = data['confidence'][:num_train]
        poses = data['odometry'][:num_train]
    elif split == 'val':
        rgbs = data['rgb'][num_train:num_train+num_val]
        depths = data['depth'][num_train:num_train+num_val]
        confidences = data['confidence'][num_train:num_train+num_val]
        poses = data['odometry'][num_train:num_train+num_val]
    else :
        rgbs = data['rgb'][num_train+num_val:]
        depths = data['depth'][num_train+num_val:]
        confidences = data['confidence'][num_train+num_val:]
        poses = data['odometry'][num_train+num_val:]

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
    split=['train','val','test']
    # train : val : test = args.num_train : 4 : remainder
    for mode in split:
        process_stray_scanner(args,data,mode)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
