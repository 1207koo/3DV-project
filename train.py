from collections import OrderedDict
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from args import args
from datasets import *
from model import *
from util import *
from d2_net.lib.model_test import D2Net
from d2_net.lib.pyramid import process_multiscale
from baselines import *

if args.wandb:
    import wandb
    wandb.init(project="d3-net", entity=args.wandb, tags=['train', args.dataset, 'dim:%d'%args.dim, 'original_dim:%d'%args.original_dim], notes=str(vars(args)))


teacher_model = D2Net(model_file='/gallery_tate/jinseo.jeong/3dcv/models/d2_tf.pth',
                      use_relu=True, use_cuda=True)

model = D3Net().to(args.device)
if len(args.devices) > 1:
    model = torch.nn.DataParallel(model, device_ids = args.devices)
print('Parameter count:', num_parameter(model))

train_loader = torch.utils.data.DataLoader(get_dataset('train'), batch_size=args.batch_size, shuffle=True, num_workers=min(8, os.cpu_count()))
val_loader = torch.utils.data.DataLoader(get_dataset('val'), batch_size=args.batch_size, shuffle=True, num_workers=min(8, os.cpu_count()))
test_loader = torch.utils.data.DataLoader(get_dataset('test'), batch_size=args.batch_size, shuffle=True, num_workers=min(8, os.cpu_count()))

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(range(args.milestones, args.epoch, args.milestones)), gamma = args.gamma)

epoch_tqdm = tqdm(range(args.epoch))
for epoch in epoch_tqdm:
    model.train()

    loss_sum = 0.0
    loss_cnt = 0

    batch_tqdm = tqdm(train_loader, leave=False)
    for batch in batch_tqdm:
        batch.to(args.device)
        optimizer.zero_grad()

        features, scores, efeatures = model(batch)
        # TODO: loss
        # TODO: make sure that images are preprocess as done by D2Net.
        gt_features = teacher_model.dense_feature_extraction(batch)
        loss = F.mse_loss(features, gt_features) #+ homography?
        #

        loss_sum += loss.item() * batch.shape[0]
        loss_cnt += batch.shape[0]

        batch_tqdm.set_description('epoch: %03d, loss: %f'%(epoch, loss_sum / loss_cnt))
        loss.backward()
        optimizer.step()

    scheduler.step()


    # results
    out_dict = OrderedDict()
    out_dict['epoch'] = epoch
    out_dict['loss'] = loss_sum / loss_cnt

    # val
    if (epoch + 1) % args.test_every == 0:
        model.eval()
        # TODO: validation code?
        with torch.no_grad():
            out_dict['val_acc'] = 0.0
    # test
    if (epoch + 1) == args.epoch:
        model.eval()
        # TODO: test code?
        with torch.no_grad():
            out_dict['test_acc'] = 0.0

    if args.save_model != '':
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        l = len(args.save_model.split('.')[-1]) + 1
        save_path = args.save_model[:-l] + '_epoch%03d'%epoch + args.save_model[-l:]
        if len(args.devices) == 1:
            torch.save(model[k].state_dict(), save_path)
        else:
            torch.save(model[k].module.state_dict(), save_path)

    # logging
    if args.wandb:
        wandb.log(out_dict)
    text = ''
    for k, v in out_dict.items():
        if text != '':
            text += ', '
        text += '%s: %s'%(str(k), str(v))
    epoch_tqdm.write(text)

# final save
if args.save_model != '':
    if len(args.devices) == 1:
        torch.save(model[k].state_dict(), args.save_model)
    else:
        torch.save(model[k].module.state_dict(), args.save_model)
