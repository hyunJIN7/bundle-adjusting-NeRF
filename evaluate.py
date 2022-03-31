import numpy as np
import os,sys,time
import torch
import importlib
import options
from util import log

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

        m.restore_checkpoint(opt)
        if opt.data.dataset in ["blender","llff","arkit","iphone"]: #TODO iphone은 테스트뷰 원래 저장안하는데 여기도 넣어볼까,EasyDict에 pose refine 없다고 에러남.
            m.evaluate_full(opt)
            m.evaluate_ckt(opt) # TODO: 잘 돌아가나 확인 필요
        m.generate_videos_synthesis(opt)

if __name__=="__main__":
    main()




    @torch.no_grad()
    def evaluate_ckt(self, opt):
        self.graph.eval()
        # 매 이터레이션마다 train pose의 ATE 평균값 계산 후 평균내서 텍스트 파일로
        #
        pose_err_list = []  # ate는 아닌데 pose,
        for ep in range(0, opt.max_iter + 1, opt.freq.ckpt):  # 5000 간격으로
            # load checkpoint (0 is random init)
            if ep != 0:
                try:
                    util.restore_checkpoint(opt, self, resume=ep)
                except:
                    continue
            # evaluate rotation/translation
            if opt.data.dataset in ["iphone"]:
                pose, pose_GT = self.get_gt_training_poses_iphone_for_eval(opt)
            else :
                pose, pose_GT = self.get_all_training_poses(opt)
            pose_aligned, self.graph.sim3 = self.prealign_cameras(opt, pose, pose_GT)
            error = self.evaluate_camera_alignment(opt, pose_aligned, pose_GT)
            rot = np.rad2deg(error.R.mean().cpu())
            trans = error.t.mean()
            pose_err_list.append(edict(ep=ep, rot=rot, trans=trans))

        ckpt_ate_fname = "{}/ckpt_quant_pose.txt".format(opt.output_path)
        with open(ckpt_ate_fname, "w") as file:
            for list in enumerate(pose_err_list):
                file.write("{} {} {}\n".format(list.ep, list.rot, list.trans))
        # nerf.py의 evla_everyiter로 접근
        super().evaluate_ckt(opt)
