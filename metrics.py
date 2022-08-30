import json
from os.path import join as pjoin
from typing import Optional

import numpy as np
from skimage.metrics import structural_similarity
import torch.nn.functional as F

from utils import load_nib


# average over first dimension:
#   projections: [lambda, u, v]
#   reconstructions: [z, x, y]


def rmse(prediction, target):
    return F.mse_loss(prediction, target).sqrt()


def nmse(prediction, target, reduce=np.mean):
    return reduce(np.sum(
        (prediction - target)**2, axis=(1, 2)) /
        np.sum(target**2, axis=(1, 2)))


def psnr(prediction, target, reduce=np.mean):
    return reduce(20*np.log10(np.prod(target.shape[1:]) *
                              np.max(target, axis=(1, 2)) /
                              np.sqrt(np.sum((prediction-target)**2,
                                             axis=(1, 2)))))


def ssim(prediction, target, reduce=np.mean):
    assert prediction.shape == target.shape
    return reduce(np.concat([
        structural_similarity(prediction[idx], target[idx])
        for idx in range(prediction.shape[0])]))


def nd2str(neighbor_diff: int):
    if neighbor_diff == 1:
        return 'nd1'
    if neighbor_diff == 2:
        return 'nd2_nd1'
    if neighbor_diff == 4:
        return 'nd4_nd2_nd1'

    raise NotImplementedError()


# TODO: update nib filenames
def projection_metrics(method: str,
                       test_subject: int,
                       neighbor_diff: Optional[int] = None):
    assert method in ['ana', 'nn2', 'nn4', 'nn8', 'nn2+ana']
    assert neighbor_diff in [None, 1, 2, 4]
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        valid_files = json_dict['test_files']
    assert test_subject < len(valid_files)

    target = load_nib(pjoin(DATA_DIRS['datasets'], f'{valid_files[test_subject]}'))
    target = np.transpose(target)

    if method == 'ana':
        prediction = load_nib(f'analytical_up{neighbor_diff}_{test_subject}.nii.gz')
    elif method.startswith('nn') and not method.endswith('ana'):
        prediction = load_nib(f'upsampled_{nd2str(neighbor_diff)}_{method}_{test_subject}.nii.gz')
    else:  # method == 'nn2+ana':
        prediction = load_nib(f'upsampled_ana_{nd2str(neighbor_diff)}_{test_subject}.nii.gz')

    prediction = np.moveaxis(prediction, -1, 0)

    ind = [
        x for x in np.arange(target.shape[0])
        if x not in np.arange(0, target.shape[0], neighbor_diff*2)
    ]

    return {
        'nmse': nmse(prediction[ind], target[ind]),
        'psnr': psnr(prediction[ind], target[ind]),
        'ssim': ssim(prediction[ind], target[ind]),
    }


def fdk_reconstruction_metrics(method: str, test_subject: int):
    assert method in ['full', 'sparse', 'ana', 'nn2', 'nn4', 'nn8', 'nn2+ana']
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        valid_files = json_dict['test_files']
    assert test_subject < len(valid_files)

    target = load_nib(pjoin(DATA_DIRS['datasets'], valid_files[test_subject]))
    target = np.transpose(target)

    prediction = load_nib(f'fdk_{method}_{test_subject}.nii.gz')
    prediction = np.transpose(prediction)

    return {
        'nmse': nmse(prediction, target, reduce=lambda _: _),
        'psnr': psnr(prediction, target, reduce=lambda _: _),
        'ssim': ssim(prediction, target, reduce=lambda _: _),
    }


def art_reconstruction_metrics(method: str, test_subject: int):
    assert method in [
        'zero', 'sparse', 'ana',
        'nn2', 'nn4', 'nn8', 'nn2+ana',
    ]
    with open('train_valid.json', 'r') as json_file:
        json_dict = json.load(json_file)
        valid_files = json_dict['test_files']
    assert test_subject < len(valid_files)

    target = load_nib(pjoin(DATA_DIRS['datasets'], valid_files[test_subject]))
    target = np.transpose(target)

    prediction = load_nib(f'art_{method}_{test_subject}.nii.gz')
    prediction = np.transpose(prediction)

    return {
        'nmse': nmse(prediction, target, reduce=lambda _: _),
        'psnr': psnr(prediction, target, reduce=lambda _: _),
        'ssim': ssim(prediction, target, reduce=lambda _: _),
    }


def all_projection_metrics():
    all_nds = [1, 2, 4]
    all_methods = ['ana', 'nn2', 'nn4', 'nn8', 'nn2+ana']
    for neighbor_diff in all_nds:
        for method in all_methods:
            metrics0 = projection_metrics(method, neighbor_diff, 0)
            metrics1 = projection_metrics(method, neighbor_diff, 1)
            print([
                neighbor_diff,
                method,
                (metrics0['nmse']+metrics1['nmse'])/2,
                (metrics0['psnr']+metrics1['psnr'])/2,
                (metrics0['ssim']+metrics1['ssim'])/2,
            ])


def all_fdk_reconstruction_metrics():
    all_methods = ['full', 'sparse', 'ana', 'nn2', 'nn4', 'nn8', 'nn2+ana']
    for method in all_methods:
        metrics0 = fdk_reconstruction_metrics(method, 0)
        metrics1 = fdk_reconstruction_metrics(method, 1)
        print([
            method,
            np.mean(np.concatenate([metrics0['nmse'], metrics1['nmse']])),
            np.mean(np.concatenate([metrics0['psnr'], metrics1['psnr']])),
            np.mean(np.concatenate([metrics0['ssim'], metrics1['ssim']])),
        ])


def all_art_reconstruction_metrics():
    all_methods = [
        'zero', 'sparse', 'ana',
        'nn2', 'nn4', 'nn8', 'nn2+ana',
    ]
    for method in all_methods:
        metrics0 = art_reconstruction_metrics(method, 0)
        metrics1 = art_reconstruction_metrics(method, 1)
        print([
            method,
            np.mean(np.concat([metrics0['nmse'], metrics1['nmse']])),
            np.mean(np.concat([metrics0['psnr'], metrics1['psnr']])),
            np.mean(np.concat([metrics0['ssim'], metrics1['ssim']])),
        ])
