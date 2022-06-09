import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args
from util import create_cnn, interpolate_dense_features

class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size
        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score

class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected

class D3Net(nn.Module):
    # detect and describe and distill
    def __init__(self, mode='train'):
        super(D3Net, self).__init__()
        self.mode = mode

        self.feature, self.feature_scale = create_cnn(args.model_config['feature'], args.model_config['feature_nonlinear'], negative=args.dim, return_scale=True)
        if self.mode == 'train':
            self.detection = SoftDetectionModule()
        else:
            self.detection = HardDetectionModule()
        self.expansion = create_cnn(args.model_config['expansion'], args.model_config['expansion_nonlinear'], kernel=1, last_dim=args.dim, negative=args.original_dim)

        if args.load_model == '':
            self.init_weight()
        else:
            self.load_state_dict(torch.load(args.load_model, map_location='cpu'))

    def init_weight(self):
        if args.model_config['feature_nonlinear'] == 'relu':
            nl = 'relu'
        elif args.model_config['feature_nonlinear'][:5] == 'lrelu':
            nl = 'leaky_relu'
        for m in self.feature.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nl)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.detection.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        if args.model_config['expansion_nonlinear'] == 'relu':
            nl = 'relu'
        elif args.model_config['expansion_nonlinear'][:5] == 'lrelu':
            nl = 'leaky_relu'
        for m in self.expansion.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nl)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch, keypoint=None, run='fse'):
        # run = subset of 'fse', f: feature, s: score, e: expansion
        # keypoint: expansion only for given keypoints (Nx2 shape, h, w order)
        out_list = []
        b = batch.shape[0]
        if keypoint is not None:
            keypoint = [kp.permute((1, 0)) / self.feature_scale - (self.feature_scale - 1) / (2 * self.feature_scale) for kp in keypoint]
        if any([c in run for c in 'fse']):
            features = self.feature(batch)
            if 'f' in run:
                out_list.append(features)
            if any([c in run for c in 'se']):
                scores = self.detection(features)
                if keypoint is not None:
                    scores = [interpolate_dense_features(keypoint[i], scores[i].unsqueeze(0))[0].squeeze(0) for i in range(b)]
                if 's' in run:
                    out_list.append(scores)
                if any([c in run for c in 'e']): # 'e' in run
                    features_ = features
                    if keypoint is not None:
                        features_ = [interpolate_dense_features(keypoint[i], features[i])[0].unsqueeze(-1) for i in range(b)]
                        # features_ = features[:, :, keypoint[0], keypoint[1]].unsqueeze(-1)
                        efeatures = [self.expansion(f_.unsqueeze(0)).squeeze(0).squeeze(-1) for f_ in features_]
                    else:
                        efeatures = self.expansion(features_)
                    if 'e' in run:
                        out_list.append(efeatures)

        return out_list