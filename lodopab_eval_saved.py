# -*- coding: utf-8 -*-
"""
Computes metrics from saved reconstructions.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import h5py
from dival import get_standard_dataset
from dival.measure import PSNR, SSIM, PSNRMeasure, SSIMMeasure
from util import lodopab_challenge_set

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
RECONSTRUCTIONS_PATH = os.path.join(RESULTS_PATH, 'reconstructions_lodopab')
GROUND_TRUTH_PATH = '/localdata/jleuschn/low_dose_dataset_challenge'
METRICS_PATH = os.path.join(RESULTS_PATH, 'metrics')

METHOD_DEFAULT = 'learnedpd'

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default=METHOD_DEFAULT)
parser.add_argument('--num_images', type=int, default=-1)
parser.add_argument('--first_image', type=int, default=0)
# parser.add_argument('--part', type=str, default='challenge')  # only challenge reconstructions are implemented
parser.add_argument('--save_metrics', action='store_true')


options = parser.parse_args()

method = options.method  # 'learnedpd', 'fbpmsdnet', 'cinn'
name = 'lodopab_{}'.format(method)
num_images = options.num_images
first_image = options.first_image
part = 'challenge'  # options.part
save_metrics = options.save_metrics

dataset = get_standard_dataset('lodopab', impl=IMPL)

ray_trafo = dataset.ray_trafo

def get_data_pair(idx, part):
    if part == 'challenge':
        observation = lodopab_challenge_set.get_observation(idx)
        file_idx = idx // lodopab_challenge_set.NUM_SAMPLES_PER_FILE
        idx_in_file = idx % lodopab_challenge_set.NUM_SAMPLES_PER_FILE
        with h5py.File(os.path.join(GROUND_TRUTH_PATH,
                                    'ground_truth_challenge_{:03d}.hdf5'
                                    .format(file_idx)), 'r') as f:
            ground_truth = f[list(f.keys())[0]][idx_in_file]
    else:
        observation, ground_truth = dataset.get_sample(idx, part)
    return observation, ground_truth

psnr_measure_data_range_1 = PSNRMeasure(data_range=1.)
ssim_measure_data_range_1 = SSIMMeasure(data_range=1.)
recos = []
psnrs = []
ssims = []
psnrs_data_range_1 = []
ssims_data_range_1 = []
if num_images == -1:
    num_images = (lodopab_challenge_set.NUM_IMAGES-first_image
                  if part == 'challenge' else
                  dataset.get_len(part)-first_image)
with tqdm(range(first_image, first_image+num_images),
          desc="eval '{}'".format(name)) as p:
    for i in p:
        obs, gt = get_data_pair(i, part)
        file_idx = i // lodopab_challenge_set.NUM_SAMPLES_PER_FILE
        idx_in_file = i % lodopab_challenge_set.NUM_SAMPLES_PER_FILE
        with h5py.File(os.path.join(
                RECONSTRUCTIONS_PATH, method, 'reco_{:03d}.hdf5'.format(
                    file_idx)), 'r') as f:
            reco = f[list(f.keys())[0]][idx_in_file]
        recos.append(reco)
        psnr = PSNR(reco, gt)
        ssim = SSIM(reco, gt)
        psnrs.append(psnr)
        ssims.append(ssim)
        psnr_data_range_1 = psnr_measure_data_range_1(reco, gt)
        ssim_data_range_1 = ssim_measure_data_range_1(reco, gt)
        psnrs_data_range_1.append(psnr_data_range_1)
        ssims_data_range_1.append(ssim_data_range_1)
        p.set_postfix({'running psnr': np.mean(psnrs),
                       'running ssim': np.mean(ssims)})
if save_metrics:
    os.makedirs(METRICS_PATH, exist_ok=True)
    # use similar style as grand-challenge.org
    metrics = {
        'case': {
            'psnr': {str(i): p for i, p in enumerate(psnrs)},
            'ssim': {str(i): s for i, s in enumerate(ssims)},
            'psnr_data_range_1': {str(i): p for i, p in
                                      enumerate(psnrs_data_range_1)},
            'ssim_data_range_1': {str(i): s for i, s in
                                      enumerate(ssims_data_range_1)}
            },
        'aggregates': {
            'psnr_max': np.max(psnrs),
            'psnr_min': np.min(psnrs),
            'psnr_mean': np.mean(psnrs),
            'psnr_std': np.std(psnrs),
            'ssim_max': np.max(ssims),
            'ssim_min': np.min(ssims),
            'ssim_mean': np.mean(ssims),
            'ssim_std': np.std(ssims),
            'psnr_data_range_1_max': np.max(psnrs_data_range_1),
            'psnr_data_range_1_min': np.min(psnrs_data_range_1),
            'psnr_data_range_1_mean': np.mean(psnrs_data_range_1),
            'psnr_data_range_1_std': np.std(psnrs_data_range_1),
            'ssim_data_range_1_max': np.max(ssims_data_range_1),
            'ssim_data_range_1_min': np.min(ssims_data_range_1),
            'ssim_data_range_1_mean': np.mean(ssims_data_range_1),
            'ssim_data_range_1_std': np.std(ssims_data_range_1),
            }
        }
    filename = os.path.join(
        METRICS_PATH, '{}_metrics.json'.format(name))
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
print()
print('mean PSNR:', np.mean(psnrs))
print('mean SSIM:', np.mean(ssims))
print('mean PSNR (data range 1):', np.mean(psnrs_data_range_1))
print('mean SSIM (data range 1):', np.mean(ssims_data_range_1))
