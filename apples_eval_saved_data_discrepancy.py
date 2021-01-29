# -*- coding: utf-8 -*-
"""
Computes data discrepancy from saved reconstructions.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import h5py
from dival.data import DataPairs
from util.apples_dataset import get_apples_dataset
from util import apples_data
from util import apples_data_test

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
RECONSTRUCTIONS_PATH = os.path.join(RESULTS_PATH, 'reconstructions')
DATA_DISCREPANCY_PATH = os.path.join(
    RESULTS_PATH, 'data_discrepancy')
DATA_DISCREPANCY_PATH_VAL = os.path.join(
    RESULTS_PATH, 'data_discrepancy_val')
SUPP_MATERIAL_PATH = 'supp_material'

NOISE_SETTING_DEFAULT = 'scattering'
NUM_ANGLES_DEFAULT = 50
METHOD_DEFAULT = 'fbpistaunet'
NUM_IMAGES_DEFAULT = 100

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)
parser.add_argument('--method', type=str, default=METHOD_DEFAULT)
parser.add_argument('--num_images', type=int, default=NUM_IMAGES_DEFAULT)
parser.add_argument('--first_image', type=int, default=0)
parser.add_argument('--secret_test_set', action='store_true')
parser.add_argument('--save_data_discrepancy', action='store_true')

options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
method = options.method  # 'learnedpd', 'fbpmsdnet', 'cinn'
name = 'apples_{}_{:02d}_{}'.format(noise_setting, num_angles, method)
num_images = options.num_images
first_image = options.first_image
secret_test_set = options.secret_test_set
save_data_discrepancy = options.save_data_discrepancy

dataset = get_apples_dataset(num_angles=num_angles,
                             noise_setting=noise_setting,
                             impl=IMPL)

ray_trafo = dataset.ray_trafo

if secret_test_set:
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
else:
    test_data = dataset.get_data_pairs_per_index(
        part='validation',
        index=list(range(first_image, first_image+num_images)))
    test_data.name = 'validation[{:d}:{:d}]'.format(
        first_image, first_image+num_images)
FIG_PATH = 'figures'
os.makedirs(FIG_PATH, exist_ok=True)
recos = []
projection_obs_mses = []
projection_ideal_obs_mses = []
with tqdm(test_data, desc="eval '{}'".format(name)) as p:
    for i, (obs, gt) in enumerate(p, start=first_image):
        if secret_test_set:
            slice_id = apples_data_test.slice_ids[test_indices[i]]
        else:
            apples_dataset = dataset.dataset
            idx = apples_dataset.indices['validation'][i]
            if apples_dataset.scattering:
                idx = apples_data.scattering_indices[idx]
            slice_id = apples_data.slice_ids[idx]
        path = os.path.join(
            RECONSTRUCTIONS_PATH,
            ('{}'.format(method) if secret_test_set else
             '{}_val'.format(method)),
            {'noisefree': 'noisefree',
             'scattering': 'scattering',
             'gaussian_noise': 'gaussian'}[noise_setting],
            'ang{:d}'.format(num_angles))
        filename = (
            'recon_noisy_{}.hdf5'.format(slice_id)
            if noise_setting == 'gaussian_noise' else
            'recon_{}.hdf5'.format(slice_id))
        with h5py.File(os.path.join(path, filename),
                       'r') as f:
            dset = f[list(f.keys())[0]]
            reco = dset[:]
        recos.append(reco)
        projection = np.asarray(ray_trafo(reco), dtype=np.float64)
        obs_float64 = np.asarray(obs, dtype=np.float64)
        ideal_obs = np.asarray(ray_trafo(gt), dtype=np.float64)
        if noise_setting == 'scattering':
            obs_float64 /= 400
        projection_obs_mse = np.mean(np.square(projection-obs_float64))
        projection_obs_mses.append(projection_obs_mse)
        projection_ideal_obs_mse = np.mean(np.square(projection-ideal_obs))
        projection_ideal_obs_mses.append(projection_ideal_obs_mse)
if save_data_discrepancy:
    os.makedirs(
        DATA_DISCREPANCY_PATH if secret_test_set else DATA_DISCREPANCY_PATH_VAL,
        exist_ok=True)
    # use similar style as grand-challenge.org
    data_discrepancy = {
        'case': {
            'projection_obs_mse': {
                str(i): m for i, m in enumerate(projection_obs_mses)},
            'projection_ideal_obs_mse': {
                str(i): m for i, m in enumerate(projection_ideal_obs_mses)},
            },
        'aggregates': {
            'projection_obs_mse_max': np.max(
                projection_obs_mses),
            'projection_obs_mse_min': np.min(
                projection_obs_mses),
            'projection_obs_mse_mean': np.mean(
                projection_obs_mses),
            'projection_obs_mse_std': np.std(
                projection_obs_mses),
            'projection_ideal_obs_mse_max': np.max(
                projection_ideal_obs_mses),
            'projection_ideal_obs_mse_min': np.min(
                projection_ideal_obs_mses),
            'projection_ideal_obs_mse_mean': np.mean(
                projection_ideal_obs_mses),
            'projection_ideal_obs_mse_std': np.std(
                projection_ideal_obs_mses),
            }
        }
    filename = os.path.join(
        DATA_DISCREPANCY_PATH if secret_test_set else DATA_DISCREPANCY_PATH_VAL,
        '{}_data_discrepancy.json'.format(name))
    with open(filename, 'w') as f:
        json.dump(data_discrepancy, f, indent=2)
print()
print('mean projection to observation mse:                   ',
      np.mean(projection_obs_mses))
print('mean projection to forward projected ground truth mse:',
      np.mean(projection_ideal_obs_mses))
