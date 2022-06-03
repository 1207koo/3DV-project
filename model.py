import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args
from util import create_cnn

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


class D3Net(nn.Module):
    # detect and describe and distill
    def __init__(self, model_file=''):
        super(D3Net, self).__init__()

        self.feature = create_cnn(args.model_config['feature'], args.model_config['feature_nonlinear'], negative=args.dim)
        self.detection = SoftDetectionModule()
        self.expansion = create_cnn(args.model_config['expansion'], args.model_config['expansion_nonlinear'], kernel=1, last_dim=args.dim, negative=args.original_dim)

        if model_file != '':
            self.load_state_dict(torch.load(model_file))

    def forward(self, batch, keypoint=None, run='fse'):
        # run = subset of 'fse', f: feature, s: score, e: expansion
        # keypoint: expansion only for given keypoints (Nx2 shape, h, w order)
        out_list = []
        if any([c in run for c in 'fse']):
            features = self.feature(batch)
            if 'f' in run:
                out_list.append(features)
            if any([c in run for c in 'se']):
                scores = self.detection(features)
                if 's' in run:
                    out_list.append(scores)
                if any([c in run for c in 'e']): # 'e' in run
                    features_ = features
                    if keypoint is not None:
                        features_ = features[:, :, keypoint[0], keypoint[1]].unsqueeze(0)
                    efeatures = self.expansion(features_)
                    if keypoint is not None:
                        efeatures = efeatures.squeeze(1)
                    if 'e' in run:
                        out_list.append(efeatures)

        return out_list