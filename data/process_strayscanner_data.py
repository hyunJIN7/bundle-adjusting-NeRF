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


DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0
np.random.seed(0)
#  conda activate StrayVisualizer-main

# python data/process_strayscanner_data.py --num_train=3 --basedir ./data/strayscanner/lab_computer_3 --depth_bound2=0.2 --depth_bound1=0.7
# python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/lab_computer_5 --depth_bound2=0.2 --depth_bound1=0.7
# python data/process_strayscanner_data.py --num_train=7 --basedir ./data/strayscanner/trashcan01_7 --depth_bound2=0.2 --depth_bound1=0.7


# python data/process_strayscanner_data.py --basedir ./data/strayscanner/trashcan01 --depth_bound2=0.2 --depth_bound1=0.7


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./data/strayscanner/computer',
                        help='input data directory')
    parser.add_argument("--num_train", type=int, default=120,
                        help='number of train data')
    parser.add_argument("--num_test", type=int, default=12,
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
    train_val_test_index = np.linspace(0, n, num_train+num_val, endpoint=False, dtype=int)
    train_val_index = np.linspace(0, n, num_train+num_val, endpoint=False, dtype=int)
    train_index = train_val_index[:-num_val]
    val_index = train_val_index[-num_val:]
    # if random sampling
    test_index = np.delete(all_index,train_val_index)
    test_index = np.random.choice(test_index,num_test,replace=False)
    test_index.sort()


    # python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/lab_desk_5ver2 --depth_bound2=0.4 --depth_bound1=1
    # train_index = np.array([0,401,1759,2500,3395])
    # val_index = np.array([61,1697,2562,2871])
    # test_index = np.array([30,61,432,1389,1728,1914,2531,2685,2963,3303,3364,3611])

    # # python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/meeting_room_5ver2 --depth_bound2=0.4 --depth_bound1=1
    # train_index = np.array([0, 122,368,395,491])
    # val_index = np.array([614,737,860,983])
    # test_index = np.array([666,1106,1351,1474,1597,1720,1843,1966,2089,2212,2335,2458])

    # # python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/pocari03_5ver2 --depth_bound2=0.4 --depth_bound1=1
    # train_index = np.array([0,89,134,179,269])
    # val_index = np.array([314,359,404,449])
    # test_index = np.array([7,74,96,111,141,150,206,262,297,309,323,486])

# python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/piano03_5ver2 --depth_bound2=0.4 --depth_bound1=1
    # piano03_5ver2
    # train_index = np.array([0,120,181,301,362])
    # val_index = np.array([422,482,543,603])
    # test_index = np.array([14,45,71,234,242,326,358,455,521,535,607,613])


    # # python data/process_strayscanner_data.py --num_train=5 --basedir ./data/strayscanner/room01_5ver2 --depth_bound2=0.4 --depth_bound1=1
    # #room01_ver2
    # train_index = np.array([125,251,376,502,753])
    # val_index = np.array([878,1004,1129,1255])
    # test_index = np.array([46,123,340,378,415,712,814,942,999,1017,1223,1323])

    # #trashcan01
    # # train_index = np.array([149,1874,2471])
    # # train_index = np.array([124,149,424,1874,2272,2471,2921])
    # train_index = np.array([149,349,1874,1548,2471])
    # val_index = np.array([2996,3021,3046,3071])
    # test_index = np.array([721,877,1200,1476,1510,1542,1722,2355,2362,2707,2992])


    # all_index = np.arange(n)
    # train_val_test_index = np.linspace(0, n, num_train+num_val+num_test, endpoint=False, dtype=int)
    # train_val_index=train_val_test_index[:-num_test]
    # train_index = train_val_index[:-num_val]
    # val_index = train_val_index[-num_val:]
    # test_index = train_val_test_index[-num_test:]
    # # if random sampling
    # test_index = np.delete(all_index,train_val_index)
    # test_index = np.random.choice(test_index,num_test,replace=False)


    rgbs = np.array(data['rgb'])
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
        skvideo.io.vwrite(os.path.join(rgb_path, str(int(pose[1])).zfill(5) + '.png'), rgb)
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
    bound2 = args.depth_bound2
    near[condi2]= torch.clamp(depth[condi2]-bound2 ,min=0)
    far[condi2] = depth[condi2]+bound2

    condi1 = confidence[..., 0] == 1
    bound1 = args.depth_bound1
    near[condi1] = torch.clamp(depth[condi1]-bound1 ,min=0)
    far[condi1] = depth[condi1]+bound1

    condi0 = confidence[..., 0] == 0
    if args.use_confi0_depth > 0:
        # consider depth
        near[condi0] = 2 #torch.clamp(depth[condi0]-0.3,max=4)
        far[condi0] = torch.clamp(depth[condi0]+0.3,min=depth_max)
    else:
        # near = 4
        near[condi0]= 4 #torch.clamp(depth[condi0]+0.3,min=4)
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
    rgb_img_path = os.path.join(args.basedir, 'rgb')
    # make_dir(rgb_img_path)
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
        # skvideo.io.vwrite(os.path.join(rgb_img_path, str(i).zfill(5) + '.jpg'), rgb)

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
