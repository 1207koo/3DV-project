import argparse
import os
import random
import numpy as np

from util import str_args, config_parse

parser = argparse.ArgumentParser(description='3D vision project parser!')

parser.add_argument("--dim", type=int, default=64, help="descriptor dimension")
parser.add_argument("--teacher", type=str, default="d2net", help="teacher model")
parser.add_argument("--original-dim", type=int, default=-1, help="descriptor dimension of teacher model (512 for D2-Net, -1 for default)")
parser.add_argument("--l2", type=bool, default=True, help="normalize feature L2=1")

parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--epoch", type=int, default=100, help="total epoch")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--milestones", type=int, default=10, help="milestones for lr scheduler")
parser.add_argument("--gamma", type=float, default=0.9, help="gamma for lr scheduler")
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer (adam, sgd)")
parser.add_argument("--scheduler", type=str, default="cosine", help="lr scheduler (cosine, multistep)")
parser.add_argument("--model-config", type=str, default="./config/model_config.txt", help="model config")

parser.add_argument("--keypoint", type=bool, default=False, help="train keypoints only (might not work for detection only methods)")
parser.add_argument("--matching", type=bool, default=False, help="train also matching with homography")
parser.add_argument("--lambda-feature", type=float, default=1.0, help="lambda for feature loss")
parser.add_argument("--lambda-score", type=float, default=1024.0, help="lambda for score loss")
parser.add_argument("--lambda-matching", type=float, default=1.0, help="lambda for matching loss")

parser.add_argument("--device", type=str, default="1", help="device(s) to use, cpu or number of gpus")
parser.add_argument("--dataset-path", type=str, default="/gallery_moma/junseo.koo/dataset", help="dataset path")
parser.add_argument("--dataset", type=str, default="COCO2014", help="dataset to use")
parser.add_argument("--img-size", type=int, default=256, help="image size")
parser.add_argument("--num-workers", type=int, default=8, help="num workers for dataset")

parser.add_argument("--test-every", type=int, default=1, help="number of epochs per test")
parser.add_argument("--save-model", type=str, default="auto", help="save model path (auto for automatic path)")
parser.add_argument("--load-model", type=str, default="", help="load model path")
parser.add_argument("--wandb", type=str, default='', help="wandb entity name")

parser.add_argument("--debug", type=bool, default=False, help="debugging mode")
parser.add_argument("--code-ver", type=str, default='0', help="code version (used for code changes during experiments)")
args = parser.parse_args()


args.teacher = args.teacher.lower().replace('-', '')
if args.original_dim == -1:
    if args.teacher == 'd2net':
        args.original_dim = 512
    elif args.teacher == 'sift':
        args.original_dim = 128
    else:
        raise NotImplementedError

if args.teacher in ['sift', 'orb', 'superpoint']:
    args.keypoint = True

if args.device == 'cpu':
    args.devices = [args.device]
else:
    args.devices = list(range(int(args.device)))
    args.device = 'cuda:0'

args.model_config = config_parse(args.model_config)

if args.wandb:
    import wandb
    wandb.init(project="d3-net", entity=args.wandb, tags=['dim:%d'%args.dim, 'teacher:%s'%args.teacher], notes=str(vars(args)))

if args.save_model == 'auto':
    if args.wandb:
        run_name = wandb.run.name
        run_idx = int(run_name.split('-')[-1])
        args.save_model = 'save/runs_wandb/run%03d_%s/run%03d_%s.pt'%(run_idx, run_name, run_idx, run_name)
    else:
        for i in range(1000):
            args.save_model = 'save/runs/run%03d'%i
            l = len(args.save_model.split('.')[-1]) + 1
            if not os.path.isdir(args.save_model):
                args.save_model = 'save/runs/run%03d/run%03d.pt'%(i, i)
                break

if args.save_model != '':
    assert not os.path.isfile(args.save_model)
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    l = len(args.save_model.split('.')[-1]) + 1
    with open(args.save_model[:-l] + '_args.txt', 'w') as f:
        f.write(str_args(args))

print(args)