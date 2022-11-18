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
from scipy.spatial.transform import Rotation
import util_vis
import camera

"""
nerf.py의 novel_pose plot 위한 코드 
"""
# python visualization_novel_view.py --expname stray
def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument("--basedir", type=str, default='./novel_poses',
                        help='input data directory')
    parser.add_argument("--expname", type=str, default='17',
                        help='experiment name')
    return parser


def generate_videos_pose(pose,pose_ref, i):# novel pose, raw pose shape : (n,3,4)
    fig = plt.figure(figsize=(10,10))

    os.makedirs("novel_poses",exist_ok=True)
    os.makedirs("sync", exist_ok=True)
    util_vis.plot_save_poses_for_oneNall_optisync(fig=fig,pose=pose[i],pose_ref=pose_ref,path="sync",ep=i)
    util_vis.plot_save_novel_poses(fig=fig, pose=pose, pose_ref=pose_ref, path="novel_poses", ep=i)
    plt.close()


def novel_view(args):
    # for GT data(optitrack)
    gt_pose_file = "./icp_17.txt"
    with open(gt_pose_file, "r") as f:  # frame.txt 읽어서
        cam_frame_lines = f.readlines()
    opti_pose = []  # time r1x y z tx r2x y z ty r3x y z tz
    for line in cam_frame_lines:
        line_data_list = line.split(' ')
        if len(line_data_list) == 0:
            continue
        pose_raw = np.reshape(line_data_list[1:], (3, 4))

        rt = pose_raw[:,:3].astype(np.float64) #@ np.array([[1,0,0],[0,1,0],[0,0,1]])
        rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
        rot = rt @ rot
        pose_raw[:,:3] = rt
        opti_pose.append(pose_raw)
    opti_pose = np.array(opti_pose, dtype=np.float64)
    opti_pose = torch.from_numpy(np.array(opti_pose)).float()
    opti_pose = opti_pose

    # for strayscanner

    pose_path = "./o17_odometry.csv"
    odometry = np.loadtxt(pose_path, delimiter=',', skiprows=1)  # , skiprows=1
    stray_poses = []
    for line in odometry:  # timestamp, frame(float ex 1.0), x, y, z, qx, qy, qz, qw
        position = line[2:5]
        quaternion = line[5:]
        T_WC = np.eye(4)
        T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        T_WC[:3, 3] = position
        T_WC = T_WC[:3]
        stray_poses.append(T_WC)
    stray_poses = torch.from_numpy(np.array(stray_poses)).float()
    # stray_poses = np.array(stray_poses, dtype=float)
    stray_poses
    generate_videos_pose(stray_poses,opti_pose,87) # novel, pose





# python visualization_novel_view.py
if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    novel_view(args)
