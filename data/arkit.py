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

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        self.raw_H,self.raw_W = 800,800
        super().__init__(opt,split)
        self.root = opt.data.root or "data/blender"
        self.path = "{}/{}".format(self.root,opt.data.scene)
        # load/parse metadata
        meta_fname = "{}/transforms_{}.json".format(self.path,split)
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]
        self.focal = 0.5*self.raw_W/np.tan(0.5*self.meta["camera_angle_x"])
        if subset: self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")

    def prefetch_all_data(self,opt):
        assert(not opt.data.augment)
        # pre-iterate through all samples and group together
        self.all = torch.utils.data._utils.collate.default_collate([s for s in self])

    def get_all_camera_poses(self,opt):
        pose_raw_all = [torch.tensor(f["transform_matrix"],dtype=torch.float32) for f in self.list]
        pose_canon_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
        return pose_canon_all

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
            pose=pose,
        )
        return sample

    def get_image(self,opt,idx):
        image_fname = "{}/{}.png".format(self.path,self.list[idx]["file_path"])
        image = PIL.Image.fromarray(imageio.imread(image_fname)) # directly using PIL.Image.open() leads to weird corruption....
        return image

    def preprocess_image(self,opt,image,aug=None):
        image = super().preprocess_image(opt,image,aug=aug)
        rgb,mask = image[:3],image[3:]
        if opt.data.bgcolor is not None:
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        return rgb

    def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],  #TODO : 여기 그 prcoss_arkit image resize해서 여기도 바꿔야해
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        """
        #Load camera intrinsics  # frane.txt -> camera intrinsics
        assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
        with open(cam_file, "r") as f:  # frame.txt 읽어서
            cam_intrinsic_lines = f.readlines()
    
        cam_intrinsics = []
        for line in cam_intrinsic_lines:
            line_data_list = line.split(',')
            if len(line_data_list) == 0:
                continue
            cam_intrinsics.append([float(i) for i in line_data_list])
            # frame.txt -> cam_instrinsic
        K = np.array([
                [cam_intrinsics[0][2], 0, cam_intrinsics[0][4]],
                [0, cam_intrinsics[0][3], cam_intrinsics[0][5]],
                [0, 0, 1]
            ])
            K[0,:] /= (ori_size[0] / size[0]) #TODO: 이거 메인 트레인 파트에서도 해줘야...
            K[1, :] /= (ori_size[1] / size[1])  #resize 전 크기가 orgin_size 이기 때문에
        """
        pose_raw = torch.tensor(self.list[idx]["transform_matrix"],dtype=torch.float32)
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose
#TODO:여기 하단은 원래 아이폰 코드로 해야하나 프레임이 뭐지
    def parse_raw_camera(self,opt,pose_raw):  #애초에 저장시킨 pose를 축 뒤집고 해놓으면 여기 처리할 필요 없지 않을까? 단 gt도 여기 기준 맞춰야하는건가...
        pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
        pose = camera.pose.compose([pose_flip,pose_raw[:3]])
        pose = camera.pose.invert(pose)
        return pose
