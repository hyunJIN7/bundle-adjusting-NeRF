import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import lpips

import util_vis
import camera

"""
nerf.py의 novel_pose plot 위한 코드 
"""
# python novel_view_test.py --expname cube
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./novel_poses',
                        help='input data directory')
    parser.add_argument("--expname", type=str, default='tri',
                        help='experiment name')
    return parser


def generate_videos_pose(pose,pose_ref):# novel pose, raw pose
    fig = plt.figure(figsize=(10,10))
    cam_path = "novel_poses"
    os.makedirs(cam_path,exist_ok=True)
    util_vis.plot_save_novel_poses(fig,pose,pose_ref=pose_ref,path=cam_path,ep=args.expname)
    plt.close()


def novel_view(args):
    rectangle_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0]]
       ).float()
    tri_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 0],]
       ).float()
    cube_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 3],
         [1, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 3],
         [1, 0, 0, 3, 0, 1, 0, 3, 0, 0, 1, 3],
         [1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1, 3]]
       ).float()
    small_cube_pose = torch.tensor(
        [[1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0],
         [1, 0, 0, 0.7, 0, 1, 0, 0,   0, 0, 1, 0],
         [1, 0, 0, 0.7, 0, 1, 0, 0.7, 0, 0, 1, 0],
         [1, 0, 0, 0,   0, 1, 0, 0.7, 0, 0, 1, 0],
         [1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0.8],
         [1, 0, 0, 0.4, 0, 1, 0, 0,   0, 0, 1, 0.8],
         [1, 0, 0, 0.4, 0, 1, 0, 0.7, 0, 0, 1, 0.8],
         [1, 0, 0, 0,   0, 1, 0, 0.7, 0, 0, 1, 0.8]]
       ).float()

    random_cube_pose = torch.tensor(
        [[1, 0, 0, 0, 0, 1, 0, 0, 1.1, 0, 1, 0],
         [1, 1, 0, 3, 0, 1, 0, 1, 0, 0, 1, -1],
         [1, 0, 1, 3, 0, 1, 0, -1, 0, 0, 1, 1],
         [1, 1, 1.3, 0, 0, 2, 3.2, 2.4, 0, 0, 2, 0],
         [1, 0, 2, 2.1, 0, 1, 0, -1, 0, 0, 1, 2],
         [1, 3, 0, 1, 0, 2, 0, 1.2, 0, 0, 1, -2],
         [1, 0, 2, 3, 0, 1, 0, 3, 2.1, 0, 1, -1],
         [2.4, 1.4, 2.1, 0, 2.5, 3, 0, 3, 0, 0, 1, 0]]
       ).float()

    # python novel_view_test.py --expname cube2
    main_pose = cube_pose
    poses = [torch.reshape(i,(3,4)) for i in main_pose] # list
    poses = torch.stack(poses)  # torch.tensor
    print(poses.shape)

    scale = 1
    # rotate novel views around the "center" camera of all poses
    idx_center = (poses - poses.mean(dim=0, keepdim=True))[..., 3].norm(dim=-1).argmin()
    pose_novel = camera.get_novel_view_poses(opt=None,pose_anchor= poses[idx_center], N=30, scale=scale)
    #novel 개수 맞춰야하나
    generate_videos_pose(pose_novel,poses) # novel, pose

# python novel_view_test.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    novel_view(args)
