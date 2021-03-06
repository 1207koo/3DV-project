from copyreg import pickle
import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import os
import time
import pickle
import torch

from scipy.io import loadmat

from tqdm import tqdm

from args import args
from model import *
from d2_net.extract_features import extract


use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

methods = ['sift', 'd2-net',]

# Change here if you want to use top K or all features.
# top_k = 2000
top_k = None 

n_i = 52
n_v = 56

dataset_path = 'hpatches-sequences-release'
if not os.path.isdir(dataset_path):
    dataset_path = 'd2_net/hpatches_sequences/hpatches-sequences-release'

save_path = '/home_klimt/junseo.koo/semester/2022-1/tai/3DV-project/save/runs_wandb'
model_list = [
    (4, 'run110_dulcet-disco-110'),
    (8, 'run109_rare-resonance-109'),
    (16, 'run108_autumn-dawn-108'),
    (32, 'run107_hopeful-field-107'),
    (64, 'run106_absurd-voice-106'),
    (128, 'run105_absurd-river-105'),
]

lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def mnn_matcher(descriptors_a, descriptors_b):
    if descriptors_a.shape[0] == 0 or descriptors_b.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()

def benchmark_features(read_feats, verbose=True):
    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names), leave=verbose):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device), 
                torch.from_numpy(descriptors_b).to(device=device)
            )
            
            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(im_idx)))
            
            pos_a = keypoints_a[matches[:, 0], : 2] 
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])
            
            if dist.shape[0] == 0:
                dist = np.array([float("inf")])
            
            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)
    
    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)
    
    return i_err, v_err, [seq_type, n_feats, n_matches]

def summary(stats, verbose=True):
    seq_type, n_feats, n_matches = stats
    if verbose:
        print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
        print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
            np.sum(n_matches) / ((n_i + n_v) * 5), 
            np.sum(n_matches[seq_type == 'i']) / (n_i * 5), 
            np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
        )
    return np.sum(n_matches) / ((n_i + n_v) * 5)

def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(dataset_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k :]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]
    return read_function

def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)
def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    if top_k is None:
        return keypoints, descriptors
    else:
        assert('scores' in mat)
        ids = np.argsort(mat['scores'][0])[-top_k :]
        return keypoints[ids, :], descriptors[ids, :]

if top_k is None:
    cache_dir = 'cache'
else:
    cache_dir = 'cache-top'
if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)

errors = {}

def test(method, verbose=True):
    output_file = os.path.join(cache_dir, method + '.npy')
    if verbose:
        print(method)
    if method == 'hesaff':
        read_function = lambda seq_name, im_idx: parse_mat(loadmat(os.path.join(dataset_path, seq_name, '%d.ppm.hesaff' % im_idx), appendmat=False))
    else:
        if method == 'delf' or method == 'delf-new':
            read_function = generate_read_function(method, extension='png')
        else:
            read_function = generate_read_function(method)
    
    elapsed = 0.
    if False:#os.path.exists(output_file):
        if verbose:
            print('Loading precomputed errors...')
        st = time.time()
        errors[method] = np.load(output_file, allow_pickle=True)
        elapsed = time.time() - st
    else:
        st = time.time()
        errors[method] = benchmark_features(read_function, verbose=verbose)
        elapsed = time.time() - st
        np.save(output_file, errors[method])
    return summary(errors[method][-1], verbose=verbose), elapsed

def savefigs(save_name):
    colors = ['orange', 'red', 'blue', 'brown', 'purple', 'green', 'yellow', 'black']
    plt_lim = [1, 10]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    for method,  color, in zip(errors.keys(),colors):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, linewidth=3, label=method)
    plt.title('Overall')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()

    plt.subplot(1, 3, 2)
    for method,  color, in zip(errors.keys(),colors):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, linewidth=3, label=method)
    plt.title('Illumination')
    plt.xlabel('threshold [px]')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.subplot(1, 3, 3)
    for method, color in zip(errors.keys(), colors,):
        i_err, v_err, _ = errors[method]
        plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, linewidth=3, label=method)
    plt.title('Viewpoint')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylim([0, 1])
    plt.gca().axes.set_yticklabels([])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.savefig('{}.pdf'.format(save_name), bbox_inches='tight', dpi=300)
    with open(os.path.join('{}.pkl'.format(save_name)), 'wb') as f:
        pickle.dump(errors, f)

def print_dict(d, save_name):
    key_list = []
    for k in sorted(d.keys()):
        if '_matches' in k:
            key_list.append(k.replace('_matches', ''))
    with open(save_name, 'w') as f:
        for k in key_list:
            f.write('%s %f %f %f\n'%(k, d[k + '_matches'], d[k + '_runtime'], d[k + '_time']))

dev = 'gpu'
if dev == 'gpu':
    test_dict = {}
    for dim, model_dir in model_list:
        model_path = os.path.join(save_path, model_dir, model_dir + '.pt')
        model = D3Net(dim=dim, load_model=model_path).to('cuda:0')
        _, test_dict['ours%d_runtime'%dim] = extract(model, '.ours%d'%dim, exist_ok=False, verbose=False)
        test_dict['ours%d_matches'%dim], test_dict['ours%d_time'%dim] = test('ours%d'%dim, verbose=False)
        extract(model, '.ours%d'%dim, exist_ok=False, verbose=False, remove_only=True)
    _, test_dict['d2net_runtime'] = extract(None, '.d2-net', exist_ok=False, verbose=False)
    test_dict['d2net_matches'], test_dict['d2net_time'] = test('d2-net', verbose=False)
    _, test_dict['sift_runtime'] = extract(None, '.sift', exist_ok=False, verbose=False)
    test_dict['sift_matches'], test_dict['sift_time'] = test('sift', verbose=False)
    savefigs('save/single')
    print_dict(test_dict, 'save/single')

    errors = {}
    test_dict = {}
    for dim, model_dir in model_list:
        model_path = os.path.join(save_path, model_dir, model_dir + '.pt')
        model = D3Net(dim=dim, load_model=model_path).to('cuda:0')
        _, test_dict['ours%d_runtime'%dim] = extract(model, '.ours%d'%dim, exist_ok=False, verbose=False, multiscale=True)
        test_dict['ours%d_matches'%dim], test_dict['ours%d_time'%dim] = test('ours%d'%dim, verbose=False)
        extract(model, '.ours%d'%dim, exist_ok=False, verbose=False, remove_only=True, multiscale=True)
    _, test_dict['d2net_runtime'] = extract(None, '.d2-net', exist_ok=False, verbose=False, multiscale=True)
    test_dict['d2net_matches'], test_dict['d2net_time'] = test('d2-net', verbose=False)
    _, test_dict['sift_runtime'] = extract(None, '.sift', exist_ok=False, verbose=False, multiscale=True)
    test_dict['sift_matches'], test_dict['sift_time'] = test('sift', verbose=False)
    savefigs('save/multi')
    print_dict(test_dict, 'save/multi')

else:
    test_dict = {}
    for dim, model_dir in model_list:
        model_path = os.path.join(save_path, model_dir, model_dir + '.pt')
        model = D3Net(dim=dim, load_model=model_path)
        _, test_dict['ours%d_runtime'%dim] = extract(model, '.ours_cpu%d'%dim, exist_ok=False, verbose=False)
        test_dict['ours%d_matches'%dim], test_dict['ours%d_time'%dim] = test('ours_cpu%d'%dim, verbose=False)
        extract(model, '.ours_cpu%d'%dim, exist_ok=False, verbose=False, remove_only=True)
    _, test_dict['d2net_runtime'] = extract(None, '.d2-net', exist_ok=False, verbose=False)
    test_dict['d2net_matches'], test_dict['d2net_time'] = test('d2-net', verbose=False)
    _, test_dict['sift_runtime'] = extract(None, '.sift', exist_ok=False, verbose=False)
    test_dict['sift_matches'], test_dict['sift_time'] = test('sift', verbose=False)
    savefigs('save/single_cpu')
    print_dict(test_dict, 'save/single_cpu')

    errors = {}
    test_dict = {}
    for dim, model_dir in model_list:
        model_path = os.path.join(save_path, model_dir, model_dir + '.pt')
        model = D3Net(dim=dim, load_model=model_path)
        _, test_dict['ours%d_runtime'%dim] = extract(model, '.ours_cpu%d'%dim, exist_ok=False, verbose=False, multiscale=True)
        test_dict['ours%d_matches'%dim], test_dict['ours%d_time'%dim] = test('ours_cpu%d'%dim, verbose=False)
        extract(model, '.ours_cpu%d'%dim, exist_ok=False, verbose=False, remove_only=True, multiscale=True)
    _, test_dict['d2net_runtime'] = extract(None, '.d2-net', exist_ok=False, verbose=False, multiscale=True)
    test_dict['d2net_matches'], test_dict['d2net_time'] = test('d2-net', verbose=False)
    _, test_dict['sift_runtime'] = extract(None, '.sift', exist_ok=False, verbose=False, multiscale=True)
    test_dict['sift_matches'], test_dict['sift_time'] = test('sift', verbose=False)
    savefigs('save/multi_cpu')
    print_dict(test_dict, 'save/multi_cpu')