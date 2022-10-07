import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import imageio
from easydict import EasyDict as edict
import json
import pickle

from . import base
import camera
from util import log,debug
from scipy.spatial.transform import Rotation
import cv2
from PIL import Image

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 192,256
        super().__init__(opt,split)
        self.root = opt.data.root or "data/iphone"
        self.path = "{}/{}".format(self.root,opt.data.scene)

        self.path_image = "{}/rgb_train_val".format(self.path) if split != "test" else "{}/rgb_test".format(self.path)
        self.list = sorted(os.listdir(self.path_image), key=lambda f: int(f.split(".")[0]))  #이미지

        intrin_file = os.path.join(self.path, 'camera_matrix.csv')
        assert os.path.isfile(intrin_file), "camera info:{} not found".format(intrin_file)
        intrinsics = np.loadtxt(intrin_file, delimiter=',')
        intrinsics = torch.from_numpy(np.array(intrinsics)).float()
        intrinsics[1,:] = intrinsics[1,:] / (1920/self.raw_H)
        intrinsics[2,:] = intrinsics[2,:] / (1920/self.raw_W)
        self.intr = intrinsics

        pose_path = "{}/odometry_train.csv".format(self.path) if split != "test" else "{}/odometry_{}.csv".format(self.path, split)
        # pose_path = os.path.join('./', pose_path)
        assert os.path.isfile(pose_path), "pose info:{} not found".format(pose_path)
        odometry = np.loadtxt(pose_path, delimiter=',')#, skiprows=1
        self.frames = odometry
        poses = []
        for line in odometry: # timestamp, frame(float ex 1.0), x, y, z, qx, qy, qz, qw
            position = line[2:5]
            quaternion = line[5:]
            T_WC = np.eye(4)
            T_WC[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
            T_WC[:3, 3] = position
            poses.append(T_WC)
        poses = torch.from_numpy(np.array(poses)).float()
        self.cam_pose = poses

        if split != "test" : #train,val
            # manually split train/val subsets
            num_val_split = int(len(self) * opt.data.val_ratio)  # len * 0.1
            self.list = self.list[:-num_val_split] if split == "train" else self.list[-num_val_split:]  # 전체에서 0.9 : 0.1 = train : test 비율
            self.cam_pose = self.cam_pose[:-num_val_split] if split == "train" else self.cam_pose[-num_val_split:]

        self.gt_pose = self.cam_pose

        self.opti_pose = self.cam_pose
        # for GT data(optitrack)
        # gt_pose_fname = "{}/opti_transforms_{}.txt".format(self.path,split)
        # gt_pose_file = os.path.join('./', gt_pose_fname)
        # if os.path.isfile(gt_pose_file): # gt file exist
        #     with open(gt_pose_file, "r") as f:  # frame.txt 읽어서
        #         cam_frame_lines = f.readlines()
        #     cam_gt_pose = []  # time r1x y z tx r2x y z ty r3x y z tz
        #     for line in cam_frame_lines:
        #         line_data_list = line.split(' ')
        #         if len(line_data_list) == 0:
        #             continue
        #         pose_raw = np.reshape(line_data_list[1:], (3, 4))
        #         cam_gt_pose.append(pose_raw)
        #     cam_gt_pose = np.array(cam_pose, dtype=float)
        #     self.opti_pose = cam_gt_pose
        # else: self.opti_pose = self.cam_pose


        if subset:
            self.list = self.list[:subset] # val 4개만
            self.cam_pose = self.cam_pose[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.cameras = self.preload_threading(opt, self.get_camera,
                                                  data_str="cameras")  # get_all_camera_poses 로 감
    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt): #기본 train할때 여기 접근해서 가져오고, data 로드할때 여기 접근
        if self.split == 'test':
            pose_raw_all = [torch.tensor(f, dtype=torch.float32) for f in self.cam_pose]
            pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        else:   #train,val initial pose I
            pose = camera.pose(t=torch.zeros(len(self),3))  # TODO :Camera 초기 포즈
        return pose

    #get_all_gt_camera_poses
    def get_all_gt_camera_poses(self,opt): # optitrack pose load
        #여기 iphone pose 평가할때 gt 데이터 로드 위해(train,val,test)
        pose_raw_all = [torch.tensor(f, dtype=torch.float32) for f in self.cam_pose]
        pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose

    #get_all_gt_camera_poses
    def get_all_optitrack_camera_poses(self,opt): # optitrack pose load
        #여기 iphone pose 평가할때 gt 데이터 로드 위해(train,val,test)
        pose_raw_all = [torch.tensor(f, dtype=torch.float32) for f in self.cam_pose]
        pose = torch.stack([self.parse_raw_camera(opt, p) for p in pose_raw_all], dim=0)
        return pose


    def __getitem__(self,idx):
        opt = self.opt
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        image = self.preprocess_image(opt,image,aug=aug)
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        intr,pose = self.preprocess_camera(opt,intr,pose,aug=aug)
        sample.update(
            image=image,
            intr=intr,
            pose=pose,  #shape (3,4)
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}".format(self.path_image,self.list[idx])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def get_camera(self,opt,idx):
        intr = self.intr
        if self.split == 'test':
            pose_raw = torch.tensor(self.cam_pose[idx],dtype=torch.float32)
            pose = self.parse_raw_camera(opt, pose_raw)
        else: pose = camera.pose(t=torch.zeros(3))
        return intr,pose

    # [right, forward, up]
    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose