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
        self.path_image = "{}/rgb_{}".format(self.path,split)
        self.list = sorted(os.listdir(self.path_image), key=lambda f: int(f.split(".")[0]))  #이미지

        intrin_file = os.path.join(self.path, 'camera_matrix.csv')
        assert os.path.isfile(intrin_file), "camera info:{} not found".format(intrin_file)
        intrinsics = np.loadtxt(intrin_file, delimiter=',')
        intrinsics = torch.from_numpy(np.array(intrinsics)).float()

        intrinsics[0, :] = intrinsics[0, :] * float(self.raw_W) / 1920.0
        intrinsics[1, :] = intrinsics[1, :] * float(self.raw_H) / 1440.0
        self.intr = intrinsics

        pose_path = "{}/odometry_{}.csv".format(self.path,split)
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

        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt, self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")
            self.gt_depth = self.preload_threading(opt, self.get_depth,data_str="depth")
            self.confidence = self.preload_threading(opt, self.get_confidence, data_str="confidence")
            self.bound = self.preload_threading(opt, self.get_bound, data_str="bound")


    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_depth(self,opt):
        depth = torch.stack([torch.tensor(f ,dtype=torch.float32) for f in self.depth])
        confidence = torch.stack([torch.tensor(f, dtype=torch.float32) for f in self.confidence])
        return depth,confidence

    #get_all_gt_camera_poses
    def get_all_gt_depth(self,opt): # optitrack pose load
        depth = torch.stack([torch.tensor(f, dtype=torch.float32) for f in self.depth])
        return depth


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

        confidence = self.confidence[idx] if opt.data.preload else self.get_confidence(opt,idx)
        gt_depth = self.gt_depth[idx] if opt.data.preload else self.get_depth(opt,idx)

        near,far = self.bound[idx] if opt.data.preload else self.get_bound(opt,idx)

        sample.update(
            confidence=confidence,
            gt_depth=gt_depth,
            image=image,
            intr=intr,
            pose=pose,  #shape (3,4)
            gt_near=near,
            gt_far=far
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
    def get_depth(self,opt,idx):
        depth_fname = "{}.npy".format(str(int(self.frames[idx][1])).zfill(5))
        depth_fname = "{}/depth_{}/{}".format(self.path,self.split,depth_fname)
        depth = torch.from_numpy(np.load(depth_fname)).float()
        return depth


    def get_confidence(self,opt,idx):
        confi_fname = "{}.npy".format(str(int(self.frames[idx][1])).zfill(5))
        confi_fname = "{}/confidence_{}/{}".format(self.path,self.split,confi_fname)
        confidence = torch.from_numpy(np.load(confi_fname))
        return confidence

    def get_bound(self,opt,idx):
        near_fname = "{}.npy".format(str(int(self.frames[idx][1])).zfill(5))
        near_fname = "{}/near_bound_{}/{}".format(self.path,self.split,near_fname)
        near = torch.from_numpy(np.load(near_fname))

        far_fname = "{}.npy".format(str(int(self.frames[idx][1])).zfill(5))
        far_fname = "{}/far_bound_{}/{}".format(self.path, self.split, far_fname)
        far = torch.from_numpy(np.load(far_fname))

        return near,far


    # [right, forward, up]
    def parse_raw_camera(self,opt,pose_raw):
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,1,1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose