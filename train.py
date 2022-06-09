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
from baselines import *
from test import test
from d2_net.extract_features import extract

if args.wandb:
    import wandb
<<<<<<< cc8a29a47a255aabce85b6b3b866cfa1aec2f542
||||||| merged common ancestors
    
=======

SIFT_DONE = False
D2_DONE = False
>>>>>>> test code


if args.teacher == 'd2net':
    if args.keypoint:
        from d2_net.lib.model_test import D2Net
        teacher_model = D2Net(model_file='./d2_net/models/d2_tf.pth', use_relu=True, use_cuda=True)
    else:
        from d2_net.lib.model import D2Net
        teacher_model = D2Net(model_file='./d2_net/models/d2_tf.pth', use_cuda=True)
    if len(args.devices) > 1:
        teacher_model = torch.nn.DataParallel(teacher_model, device_ids = args.devices)
elif args.teacher == 'sift':
    teacher_model = None
else:
    raise NotImplementedError

model = D3Net().to(args.device)
if len(args.devices) > 1:
    model = torch.nn.DataParallel(model, device_ids = args.devices)
print('Parameter count:', num_parameter(model))

train_loader = torch.utils.data.DataLoader(get_dataset('train'), batch_size=args.batch_size, shuffle=True, num_workers=min(args.num_workers, os.cpu_count()), pin_memory=True)
# val_loader = torch.utils.data.DataLoader(get_dataset('val'), batch_size=args.batch_size, shuffle=False, num_workers=min(args.num_workers, os.cpu_count()), pin_memory=True)
# test_loader = torch.utils.data.DataLoader(get_dataset('test'), batch_size=args.batch_size, shuffle=False, num_workers=min(args.num_workers, os.cpu_count()), pin_memory=True)

lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
length = len(train_loader)
if args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.milestones * length)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = list(np.arange(args.milestones, args.epoch, args.milestones) * length), gamma = args.gamma)

epoch_tqdm = tqdm(range(args.epoch), desc='epoch_tqdm')
for epoch in epoch_tqdm:
    model.train()

    loss_sum = 0.0
    feature_loss_sum = 0.0
    score_loss_sum = 0.0
    matching_loss_sum = 0.0
    loss_cnt = 0

    batch_tqdm = tqdm(train_loader, desc='batch', leave=False)
    for img in batch_tqdm:
        b = img.shape[0]
        img = img.to(args.device, non_blocking=True)
        optimizer.zero_grad()

        # forward
        if args.keypoint:
            gt_keypoints, gt_features = extract_teacher(img, teacher_model)
            features, scores, efeatures = model(img, keypoint=gt_keypoints)
        else:
            features, scores, efeatures = model(img)

        # feature loss
        if args.keypoint:
            if args.l2:
                feature_loss = torch.stack([F.mse_loss(F.normalize(efeatures[i]), F.normalize(gt_features[i])) for i in range(b)], dim=0).mean()
            else:
                feature_loss = torch.stack([F.mse_loss(efeatures[i], gt_features[i]) for i in range(b)], dim=0).mean()
        else:
            with torch.no_grad():
                if len(args.devices) > 1:
                    gt_features = teacher_model.module.dense_feature_extraction(img)
                else:
                    gt_features = teacher_model.dense_feature_extraction(img)
            if args.l2:
                feature_loss = F.mse_loss(F.normalize(interpolate(efeatures, gt_features.shape[2:])), F.normalize(gt_features))
            else:
                feature_loss = F.mse_loss(interpolate(efeatures, gt_features.shape[2:]), gt_features)
        
        # score loss
        if args.keypoint:
            gt_scores = [torch.ones_like(scores[i]) / scores[i].shape[0] for i in range(b)]
            score_loss = torch.stack([F.binary_cross_entropy(scores[i], gt_scores[i]) - F.binary_cross_entropy(gt_scores[i], gt_scores[i]) for i in range(b)], dim=0).mean()
        else:
            with torch.no_grad():
                if len(args.devices) > 1:
                    gt_scores = teacher_model.module.detection(gt_features)
                else:
                    gt_scores = teacher_model.detection(gt_features)
            score_loss = F.binary_cross_entropy(interpolate(scores, gt_scores.shape[1:]), gt_scores)
            score_loss -= F.binary_cross_entropy(gt_scores, gt_scores)
        
        # matching loss
        # TODO: homography?
        if args.matching:
            raise NotImplementedError
            matching_loss = 0.0

        feature_loss *= args.lambda_feature
        score_loss *= args.lambda_score
        if args.matching:
            matching_loss *= args.lambda_matching
        loss = feature_loss + score_loss
        if args.matching:
            loss += args.matching_loss

        loss_sum += loss.item() * img.shape[0]
        feature_loss_sum += feature_loss.item() * img.shape[0]
        score_loss_sum += score_loss.item() * img.shape[0]
        if args.matching:
            matching_loss_sum += matching_loss.item() * img.shape[0]
        loss_cnt += img.shape[0]
        if args.matching:
            batch_tqdm.set_description('epoch%03d > loss: %f, feature: %f, score: %f, matching: %f, lr: %f'%(epoch, loss_sum / loss_cnt, feature_loss_sum / loss_cnt, score_loss_sum / loss_cnt, matching_loss_sum / loss_cnt, float(scheduler.get_last_lr()[0])))
        else:
            batch_tqdm.set_description('epoch%03d > loss: %f, feature: %f, score: %f, lr: %f'%(epoch, loss_sum / loss_cnt, feature_loss_sum / loss_cnt, score_loss_sum / loss_cnt, float(scheduler.get_last_lr()[0])))

        loss.backward()
        optimizer.step()
        scheduler.step()


    # results
    out_dict = OrderedDict()
    out_dict['epoch'] = epoch
    out_dict['loss'] = loss_sum / loss_cnt
    out_dict['feature_loss'] = feature_loss_sum / loss_cnt
    out_dict['score_loss'] = score_loss_sum / loss_cnt
    if args.matching:
        out_dict['matching_loss'] = matching_loss_sum / loss_cnt

    # val
    if (epoch + 1) % args.test_every == 0:
        model.eval()
        # TODO: validation code?
        with torch.no_grad():
            out_dict['val_acc'] = 0.0
        if 'best_val_acc' not in out_dict.keys():
            out_dict['best_val_acc'] = out_dict['val_acc']
        elif out_dict['best_val_acc'] < out_dict['val_acc']:
            out_dict['best_val_acc'] = out_dict['val_acc']
    # test
    if (epoch + 1) == args.epoch:
        model.eval()
        # TODO: test code?
        if not SIFT_DONE:
            extract(None, '.sift')
            out_dict['test_sift_matches'] = test('sift')
            SIFT_DONE = True
        if not D2_DONE:
            extract(None, '.d2-net')
            out_dict['test_d2net_matches'] = test('d2-net')
        
        #extract(model, '.ours')
        #with torch.no_grad():          
        #    out_dict['test_acc'] = test('ours')

    if args.save_model != '':
        l = len(args.save_model.split('.')[-1]) + 1
        save_path = args.save_model[:-l] + '_epoch%03d'%epoch + args.save_model[-l:]
        if len(args.devices) == 1:
            torch.save(model.state_dict(), save_path)
        else:
            torch.save(model.module.state_dict(), save_path)

    # logging
    text = ''
    for k, v in out_dict.items():
        if text != '':
            text += ', '
        text += '%s: %s'%(str(k), str(v))
    epoch_tqdm.write(text)
    if args.wandb:
        wandb.log(out_dict)

    # update scheduler
    if (epoch + 1) % args.milestones == 0 and args.scheduler == 'cosine':
        lr *= args.gamma
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.milestones * length)

# final save
if args.save_model != '':
    if len(args.devices) == 1:
        torch.save(model.state_dict(), args.save_model)
    else:
        torch.save(model.module.state_dict(), args.save_model)
