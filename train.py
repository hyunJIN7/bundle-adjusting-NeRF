import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# python3 train.py --group=strayscanner or iphone
#                   --model=barf
#                   --yaml=barf_strayscanner or iphone
#                   --name=statue_2 --data.scene=statue --barf_c2f=[0.1,0.5]
#
def main():

    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):

        model = importlib.import_module("model.{}".format(opt.model))
        m = model.Model(opt)

        m.load_dataset(opt)
        m.build_networks(opt)
        m.setup_optimizer(opt)
        m.restore_checkpoint(opt)
        m.setup_visualizer(opt)

        m.train(opt)

if __name__=="__main__":
    main()
