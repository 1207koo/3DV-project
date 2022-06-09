import argparse

import numpy as np

import imageio

import torch
import cv2

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from .lib.model_test import D2Net
from .lib.utils import preprocess_image
from .lib.pyramid import process_multiscale

MODEL_FILE = 'models/d2_tf.pth'
USE_RELU = True
MAX_EDGE = 1600
MAX_SUM_EDGES = 2800
PREPROCESSING = 'caffe'
MULTISCALE = False
OUTPUT_TYPE = 'npz'

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def extract(model=None, output_extension='.d2-net'):
    # Creating CNN model
    if output_extension == '.d2-net':
        model = D2Net(
            model_file=MODEL_FILE,
            use_relu=USE_RELU,
            use_cuda=use_cuda
        )
    elif output_extension == '.ours':
        assert model is not None, "specify the model"
    else:
        model = cv2.xfeatures2d.SIFT_create()

    # Process the file
    with open('image_list_hpatches_sequences.txt', 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines, total=len(lines)):
        path = line.strip()

        image = imageio.imread(path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) > MAX_EDGE:
            resized_image = scipy.misc.imresize(
                resized_image,
                MAX_EDGE / max(resized_image.shape)
            ).astype('float')
        if sum(resized_image.shape[: 2]) > MAX_SUM_EDGES:
            resized_image = scipy.misc.imresize(
                resized_image,
                MAX_SUM_EDGES / sum(resized_image.shape[: 2])
            ).astype('float')

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing=PREPROCESSING
        )

        if output_extension == '.sift':
            keypoints, descriptors =  model.detectAndCompute(image, None)
            keypoints = np.asarray([[*kp.pt, 1] for kp in keypoints])
            scores = np.zeros(len(keypoints))
        elif output_extension == '.ours':
            # TODO
            features, scores, efeatures = model(image)
            keypoints = features
            descriptors = efeatures
            raise NotImplementedError
        else:
            with torch.no_grad():
                if MULTISCALE:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=device
                        ),
                        model
                    )
                else:
                    keypoints, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=device
                        ),
                        model,
                        scales=[1]
                    )
                # i, j -> u, v
                keypoints = keypoints[:, [1, 0, 2]]

            # Input image coordinates
            keypoints[:, 0] *= fact_i
            keypoints[:, 1] *= fact_j

        if OUTPUT_TYPE == 'npz':
            with open(path + output_extension, 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=keypoints,
                    scores=scores,
                    descriptors=descriptors
                )
        elif OUTPUT_TYPE == 'mat':
            with open(path + output_extension, 'wb') as output_file:
                scipy.io.savemat(
                    output_file,
                    {
                        'keypoints': keypoints,
                        'scores': scores,
                        'descriptors': descriptors
                    }
                )
        else:
            raise ValueError('Unknown output type.')

