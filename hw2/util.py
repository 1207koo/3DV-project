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

def process_multiscale_d3(image, model, scales=[.5, 1, 2]):
    b, _, h_init, w_init = image.size()
    device = image.device
    assert(b == 1)

    all_keypoints = torch.zeros([3, 0])
    all_descriptors = torch.zeros([
        model.feature_dim, 0
    ])
    all_scores = torch.zeros(0)

    previous_dense_features = None
    banned = None
    for idx, scale in enumerate(scales):
        current_image = F.interpolate(
            image, scale_factor=scale,
            mode='bilinear', align_corners=True
        )
        _, _, h_level, w_level = current_image.size()

        dense_features = model.feature(current_image)
        del current_image

        _, _, h, w = dense_features.size()

        # Sum the feature maps.
        if previous_dense_features is not None:
            dense_features += F.interpolate(
                previous_dense_features, size=[h, w],
                mode='bilinear', align_corners=True
            )
            del previous_dense_features

        # Recover detections.
        detections = model.detection_hard(dense_features)
        if banned is not None:
            banned = F.interpolate(banned.float(), size=[h, w]).bool()
            detections = torch.min(detections, ~banned)
            banned = torch.max(
                torch.max(detections, dim=1)[0].unsqueeze(1), banned
            )
        else:
            banned = torch.max(detections, dim=1)[0].unsqueeze(1)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections

        # Recover displacements.
        displacements = model.localization(dense_features)[0].cpu()
        displacements_i = displacements[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        displacements_j = displacements[
            1, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ]
        del displacements

        mask = torch.min(
            torch.abs(displacements_i) < 0.5,
            torch.abs(displacements_j) < 0.5
        )
        fmap_pos = fmap_pos[:, mask]
        valid_displacements = torch.stack([
            displacements_i[mask],
            displacements_j[mask]
        ], dim=0)
        del mask, displacements_i, displacements_j

        fmap_keypoints = fmap_pos[1 :, :].float() + valid_displacements
        del valid_displacements

        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.t().to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            continue
        fmap_pos = fmap_pos[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        # TODO: auto scaling?
        keypoints = upscale_positions(fmap_keypoints, scaling_steps=3)
        del fmap_keypoints

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        keypoints[0, :] *= h_init / h_level
        keypoints[1, :] *= w_init / w_level

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) * 1 / scale,
        ], dim=0)

        scores = dense_features[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ].cpu() / (idx + 1)
        del fmap_pos

        all_keypoints = torch.cat([all_keypoints, keypoints], dim=1)
        all_descriptors = torch.cat([all_descriptors, descriptors], dim=1)
        all_scores = torch.cat([all_scores, scores], dim=0)
        del keypoints, descriptors

        previous_dense_features = dense_features
        del dense_features
    del previous_dense_features, banned

    keypoints = all_keypoints.t().numpy()
    del all_keypoints
    scores = all_scores.numpy()
    del all_scores
    descriptors = all_descriptors.t().numpy()
    del all_descriptors
    return keypoints, scores, descriptors
