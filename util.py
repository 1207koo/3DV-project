import os
import numpy as np
import torch
import torch.nn as nn

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
        config[line[:l]] = eval(line[l+1:])
    
    return config

def num_parameter(model):
    return np.sum([p.numel() for p in model.parameters()])

def create_cnn(model_config, nonlinear='relu', kernel=3, last_dim=3, negative=512):
    layers = []
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
                layers.append(nn.LeakuReLU(float(nonlinear[6:])))
        elif type(l) == str:
            if l[0] == 'M':
                if len(l) == 0:
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    s = int(l[1:])
                    layers.append(nn.MaxPool2d(s, s))
            elif l[0] == 'B':
                layers.append(nn.BatchNorm2d(last_dim))
    return nn.Sequential(layers)