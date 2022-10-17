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


DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

#  conda activate StrayVisualizer-main
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/computer01  --num_train=200
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/computer01_120  --num_train=120
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/dinosaur
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/xyz
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/xyz2 --num_train=200
# python data/process_strayscanner_data.py --basedir ./data/strayscanner/hallway02 --num_train=200
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--num_train", type=int, default=130,
                        help='number of train data')
    parser.add_argument("--num_test", type=int, default=20,
                        help='number of train data')

    parser.add_argument("--near_range", type=int, default=2,
                        help='near near range')
    parser.add_argument("--far_range", type=int, default=6,
                        help='far near range')
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
    near_path = "{}/near_bound_{}".format(args.basedir, split)
    far_path = "{}/far_bound_{}".format(args.basedir, split)
    make_dir(rgb_path)
    make_dir(depth_path)
    make_dir(confidence_path)
    make_dir(near_path)
    make_dir(far_path)

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

    nears, fars = precompute_depth_sampling(args.near_range,args.far_range, depths, confidences) #(N,H,W)

    pose_fname = "{}/odometry_{}.csv".format(args.basedir, split)
    pose_file = open(pose_fname,'w')#,newline=','
    wr = csv.writer(pose_file)
    for i, (rgb, depth, confidence, pose,near,far) in enumerate(zip(rgbs, depths,confidences,poses,nears,fars)):
        #pose :  timestamp, frame, x, y, z, qx, qy, qz, qw
        cv2.imwrite(os.path.join(rgb_path, str(int(pose[1])).zfill(5) + '.png'), rgb)
        np.save(os.path.join(depth_path, str(int(pose[1])).zfill(5) + '.npy'), depth)
        np.save(os.path.join(confidence_path, str(int(pose[1])).zfill(5) + '.npy'), confidence)
        np.save(os.path.join(near_path, str(int(pose[1])).zfill(5) + '.npy'), near)
        np.save(os.path.join(far_path, str(int(pose[1])).zfill(5) + '.npy'), far)
        wr.writerow(pose)
    pose_file.close()



def precompute_depth_sampling(origin_near,origin_far,depth,confidence):
    #TODO : 지금 기준은 confidence , 성능 구리면 depth 값 기준으로도 더 조건 추가 4.5 이상이면 해보고 별로면
    depth_min, depth_max = origin_near, origin_far
    # [N,H,W]
    depth = torch.tensor(depth)
    confidence = torch.tensor(confidence)

    depth = depth[...,None]  #[N,H,W,1]
    confidence = confidence[..., None] #[N,H,W,1]
    near = torch.ones_like(depth) #[N,H,W,1]
    far = torch.ones_like(depth)

    condi2 = confidence[..., 0] == 2 #[N,H,W]
    near[condi2]= torch.clamp(depth[condi2]-0.3 ,min=0)
    far[condi2] = depth[condi2]+0.3

    condi1 = confidence[..., 0] == 1
    near[condi1] = torch.clamp(depth[condi1]-0.8 ,min=0)
    far[condi1] = depth[condi1]+0.8

    condi0 = confidence[..., 0] == 0
    near[condi0]= torch.clamp(depth[condi0]-0.3,max=4)
    far[condi0] = torch.clamp(depth[condi0]+0.3,min=depth_max)

    test_near , test_far = near[...,0],far[...,0]
    test_depth = depth[...,0]
    return near[...,0],far[...,0]  #[B,H*W]


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

    split = ['train', 'val', 'test'] # conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    for mode in split:
        process_stray_scanner(args,data,mode)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)
