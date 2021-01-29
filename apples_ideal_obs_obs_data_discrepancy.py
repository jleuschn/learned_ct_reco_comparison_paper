# -*- coding: utf-8 -*-
"""
Computes data discrepancy between noise-free and noisy observations.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from dival import get_standard_dataset
from dival.data import DataPairs
from util import apples_data_test

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
GROUND_TRUTH_PATH = '/localdata/jleuschn/low_dose_dataset_challenge'
DATA_DISCREPANCY_PATH = os.path.join(RESULTS_PATH, 'data_discrepancy')
SUPP_MATERIAL_PATH = 'supp_material'

NOISE_SETTING_DEFAULT = 'gaussian_noise'
NUM_ANGLES_DEFAULT = 50

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)
parser.add_argument('--num_images', type=int, default=-1)
parser.add_argument('--first_image', type=int, default=0)
parser.add_argument('--save_data_discrepancy', action='store_true')


options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
name = 'apples_{}_{:02d}_ideal_obs_obs'.format(noise_setting, num_angles)
num_images = options.num_images
first_image = options.first_image
save_data_discrepancy = options.save_data_discrepancy

ray_trafo = apples_data_test.get_ray_trafo(num_angles, impl=IMPL)

with open(os.path.join(
        SUPP_MATERIAL_PATH, 'apples', 'test_samples.json'), 'r') as f:
    test_indices = json.load(f)['test_indices']
observations = [
    apples_data_test.get_observation(
        idx, noise_setting=noise_setting, num_angles=num_angles)
    for idx in test_indices[first_image:first_image+num_images]]
ground_truth = [
    apples_data_test.get_ground_truth(idx)
    for idx in test_indices[first_image:first_image+num_images]]
test_data = DataPairs(observations=observations, ground_truth=ground_truth,
                      name='secret_test_set[{:d}:{:d}]'.format(
                          first_image, first_image+num_images))

ideal_obs_obs_mses = []
if num_images == -1:
    num_images = len(test_data)-first_image
with tqdm(test_data, desc="eval '{}'".format(name)) as p:
    for i, (obs, gt) in enumerate(p, start=first_image):
        obs_float64 = np.asarray(obs, dtype=np.float64)
        if noise_setting == 'scattering':
            obs_float64 /= 400
        ideal_obs = np.asarray(ray_trafo(gt), dtype=np.float64)

        ideal_obs_obs_mse = np.mean(np.square(obs_float64-ideal_obs))
        ideal_obs_obs_mses.append(
            ideal_obs_obs_mse)

if save_data_discrepancy:
    os.makedirs(DATA_DISCREPANCY_PATH, exist_ok=True)
    # use similar style as grand-challenge.org
    data_discrepancy = {
        'case': {
            'ideal_obs_obs_mse': {
                str(i): m
                for i, m in enumerate(ideal_obs_obs_mses)},
            },
        'aggregates': {
            'ideal_obs_obs_mse_max': np.max(
                ideal_obs_obs_mses),
            'ideal_obs_obs_mse_min': np.min(
                ideal_obs_obs_mses),
            'ideal_obs_obs_mse_mean': np.mean(
                ideal_obs_obs_mses),
            'ideal_obs_obs_mse_std': np.std(
                ideal_obs_obs_mses),
            }
        }
    filename = os.path.join(
        DATA_DISCREPANCY_PATH, '{}_data_discrepancy.json'.format(name))
    with open(filename, 'w') as f:
        json.dump(data_discrepancy, f, indent=2)
print()
print('mean (obs - obs_ideal)**2',
      np.mean(ideal_obs_obs_mses))
