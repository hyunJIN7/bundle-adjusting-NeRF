import numpy as np
import os,sys,time
import torch
import importlib
import options
from util import log


# @torch.no_grad()
# def generate_pose_image_everyiter(self,opt):
#     # 매 이터레이션마다 train pose의 ATE 평균값 계산 후 평균내서 텍스트 파일로
#     ate_fname = "{}/ATE_pose.txt".format(opt.output_path)
#     ep_list = []
#     for ep in range(0, opt.max_iter + 1, opt.freq.ckpt):  # 5000 간격으로
#         # load checkpoint (0 is random init)
#         if ep != 0:
#             try:
#                 util.restore_checkpoint(opt, self, resume=ep)
#             except:
#                 continue
#         # get the camera poses
#         pose, pose_ref = self.get_all_training_poses(opt)
#         if opt.data.dataset in ["arkit", "blender", "llff"]:
#             pose_aligned, _ = self.prealign_cameras(opt, pose, pose_ref)
#             pose_aligned, pose_ref = pose_aligned.detach().cpu(), pose_ref.detach().cpu()
#             dict(
#                 blender=util_vis.plot_save_poses_blender,
#                 llff=util_vis.plot_save_poses,
#                 arkit=util_vis.plot_save_poses_blender,  # TODO : 여기가 그 블랜터랑 포즈 결과 비주얼 다른 곳
#             )[opt.data.dataset](opt, fig, pose_aligned, pose_ref=pose_ref, path=cam_path, ep=ep)
#         else:
#             pose = pose.detach().cpu()
#             util_vis.plot_save_poses(opt, fig, pose, pose_ref=None, path=cam_path, ep=ep)
#         ep_list.append(ep)


def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for evaluating NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt,eval_split="test")
        m.build_networks(opt)

        if opt.model=="barf":
            m.generate_videos_pose(opt)
            #TODO : rgb image   --> generate_videos_synthesis

        #여기다 매 이터마다 포즈랑 이미지 저장하게 뭐 만들던가 gen_video,eval_full에서 이미지랑 포즈 만드는거 참고해서
        # m.generate_pose_image_everyiter(opt)

        m.restore_checkpoint(opt)
        if opt.data.dataset in ["blender","llff","arkit","iphone"]: #TODO iphone은 테스트뷰 원래 저장안하는데 여기도 넣어볼까,EasyDict에 pose refine 없다고 에러남.
            m.evaluate_full(opt)
        m.generate_videos_synthesis(opt)

if __name__=="__main__":
    main()
