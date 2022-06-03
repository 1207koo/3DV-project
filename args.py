import argparse
import os
import random
import numpy as np

from util import config_parse

parser = argparse.ArgumentParser(description='3D vision project parser!')

parser.add_argument("--dim", type=int, default=32, help="descriptor dimension")
parser.add_argument("--original-dim", type=int, default=512, help="descriptor dimension or original model (512 for D2-Net)")

parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--milestones", type=int, default=10, help="milestones for lr scheduler")
parser.add_argument("--gamma", type=float, default=0.9, help="gamma for lr scheduler")
parser.add_argument("--model-config", type=str, default="./config/model_config", help="model config")

parser.add_argument("--device", type=str, default="1", help="device(s) to use, cpu or number of gpus")
parser.add_argument("--dataset-path", type=str, default="/gallery_moma/junseo.koo/dataset", help="dataset path")
parser.add_argument("--dataset", type=str, default="COCO2014", help="dataset to use")
parser.add_argument("--img-size", type=int, default=256, help="image size")

parser.add_argument("--test-every", type=int, default=5, help="number of epochs per test")
parser.add_argument("--save-model", type=str, default="", help="save model path")
parser.add_argument("--load-model", type=str, default="", help="load model path")
parser.add_argument("--wandb", type=str, default='', help="wandb entity name")

args = parser.parse_args()


if args.device[:5] == "cuda:" and len(args.device) > 5:
    args.devices = []
    for i in range(len(args.device) - 5):
        args.devices.append(int(args.device[i+5]))
    args.device = args.device[:6]
else:
    args.devices = [args.device]

if args.device == 'cpu':
    args.devices = [args.device]
else:
    args.devices = []
    for i in range(int(args.device)):
        args.devices.append('cuda:%d'%i)

args.model_config = config_parse(args.model_config)