import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args


if args.wandb:
    import wandb
    wandb.init(project="d3-net", entity=args.wandb, tags=['train', args.dataset, 'dim:%d'%args.dim, 'original_dim:%d'%args.original_dim], notes=str(vars(args)))