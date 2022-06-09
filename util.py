import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def str_args(args):
    text = ''
    for k, v in vars(args).items():
        text += '%s=%s\n'%(str(k), str(v))
    return text

def config_parse(config_path):
    assert os.path.isfile(config_path), '%s config file not exists'%config_path
    with open(config_path, 'r') as f:
        lines = f.readlines()
        
    config = {}
    parsed_line = []
    for line in lines:
        line = line.replace('\n', '').split('#')[0]
        if len(line) == 0:
            continue

        if '=' in line or ':' in line:
            parsed_line.append(line)
        elif len(parsed_line) > 0:
            parsed_line[-1] = parsed_line[-1] + line

    for line in parsed_line:
        s = '='
        if '=' in line:
            s = '='
        elif ':' in line:
            s = ':'
        else:
            continue
        l = len(line.split(s)[0])
        config[line[:l].replace(' ', '')] = eval(line[l+1:])
    
    return config

def num_parameter(model):
    return np.sum([p.numel() for p in model.parameters()])

def create_cnn(model_config, nonlinear='relu', kernel=3, last_dim=3, negative=512, return_scale = False):
    layers = []
    scale = 1
    for l in model_config:
        if type(l) == int:
            if l == -1:
                layers.append(nn.Conv2d(last_dim, negative, kernel, padding=kernel//2))
                continue # no non-linear
            layers.append(nn.Conv2d(last_dim, l, kernel, padding=kernel//2))
            last_dim = l
            if nonlinear == 'relu':
                layers.append(nn.ReLU())
            elif nonlinear[:5] == 'lrelu':
                layers.append(nn.LeakyReLU(float(nonlinear[6:])))
        elif type(l) == str:
            if l[0] == 'M':
                if len(l) == 1:
                    layers.append(nn.MaxPool2d(2, 2))
                    scale *= 2
                else:
                    s = int(l[1:])
                    layers.append(nn.MaxPool2d(s, s))
                    scale *= s
            elif l[0] == 'B':
                layers.append(nn.BatchNorm2d(last_dim))
    if return_scale:
        return nn.Sequential(*layers), scale
    return nn.Sequential(*layers)

def upscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = pos * 2 + 0.5
    return pos


def downscale_positions(pos, scaling_steps=0):
    for _ in range(scaling_steps):
        pos = (pos - 0.5) / 2
    return pos


def interpolate_dense_features(pos, dense_features, return_corners=False):
    device = pos.device

    ids = torch.arange(0, pos.size(0), device=device)

    _, h, w = dense_features.size()

    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    ids = ids[valid_corners]
    # if ids.size(0) == 0:
    #     raise EmptyTensorError

    # Interpolation
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    descriptors = (
        w_top_left * dense_features[:, i_top_left, j_top_left] +
        w_top_right * dense_features[:, i_top_right, j_top_right] +
        w_bottom_left * dense_features[:, i_bottom_left, j_bottom_left] +
        w_bottom_right * dense_features[:, i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(-1, 1), j.view(-1, 1)], dim=1)

    if not return_corners:
        return [descriptors, pos, ids]
    else:
        corners = torch.stack([
            torch.stack([i_top_left, j_top_left], dim=0),
            torch.stack([i_top_right, j_top_right], dim=0),
            torch.stack([i_bottom_left, j_bottom_left], dim=0),
            torch.stack([i_bottom_right, j_bottom_right], dim=0)
        ], dim=0)
        return [descriptors, pos, ids, corners]

def interpolate(feature, size):
    if feature.shape[-len(size):] == size:
        return feature
    if len(feature.shape) == 2 and len(size) == 2:
        return F.interpolate(feature.unsqueeze(0).unsqueeze(1), size=size, mode='bilinear', align_corners=True).squeeze(1).squeeze(0)
    if len(feature.shape) == 3 and len(size) == 2:
        return F.interpolate(feature.unsqueeze(1), size=size, mode='bilinear', align_corners=True).squeeze(1)
    return F.interpolate(feature, size=size, mode='bilinear', align_corners=True)