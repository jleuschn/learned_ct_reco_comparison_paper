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
from dival import get_standard_dataset
from dival.util.constants import MU_MAX
from util import lodopab_challenge_set
from scipy.special import factorial

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
RECONSTRUCTIONS_PATH = os.path.join(RESULTS_PATH, 'reconstructions_lodopab')
GROUND_TRUTH_PATH = '/localdata/jleuschn/low_dose_dataset_challenge'
DATA_DISCREPANCY_PATH = os.path.join(RESULTS_PATH, 'data_discrepancy')

METHOD_DEFAULT = 'fbpmsdnet'

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default=METHOD_DEFAULT)
parser.add_argument('--num_images', type=int, default=-1)
parser.add_argument('--first_image', type=int, default=0)
# parser.add_argument('--part', type=str, default='challenge')  # only challenge reconstructions are implemented
parser.add_argument('--save_data_discrepancy', action='store_true')


options = parser.parse_args()

method = options.method  # 'learnedpd', 'fbpmsdnet', 'cinn'
name = 'lodopab_{}'.format(method)
num_images = options.num_images
first_image = options.first_image
part = 'challenge'  # options.part
save_data_discrepancy = options.save_data_discrepancy

dataset = get_standard_dataset('lodopab', impl=IMPL)

ray_trafo = dataset.ray_trafo

def poisson_loss(y_pred, y_true, photons_per_pixel=4096, mu_max=MU_MAX,
                 include_const=False):
    """
    Loss corresponding to Poisson regression (cf. [2]_) for post-log CT data.
    The default parameters are based on the LoDoPaB dataset creation
    (cf. [3]_).

    :Original authors:
        Sören Dittmer <sdittmer@math.uni-bremen.de>

    Parameters
    ----------
    y_pred : array
        Predicted observation (post-log, normalized by `mu_max`).
        Each entry determines a parameter of an independent Poisson model:
        ``E(N_1[:]) = N_0 exp(-y_pred[:] * mu_max)``.
    y_true : array
        True observation (post-log, normalized by `mu_max`).
        The loss evaluates how likely this observation is under the parameters
        specified via `y_pred`.
    photons_per_pixel : int or float, optional
        Mean number of photons per detector pixel for an unattenuated beam.
        Default: `4096`.
    mu_max : float, optional
        Normalization factor, by which `y_pred` and `y_true` have
        been divided (this function will multiply by it accordingly).
        Default: ``dival.util.constants.MU_MAX``.
    include_const : bool, optional
        Whether to include the constant log-factorial part.
        Default: ``False``.

    References
    ----------
    .. [2] https://en.wikipedia.org/wiki/Poisson_regression
    .. [3] https://github.com/jleuschn/lodopab_tech_ref/blob/master/create_dataset.py
    """

    def get_photons(y):
        y = np.exp(-y * mu_max) * photons_per_pixel
        return y

    def get_photons_log(y):
        y = -y * mu_max + np.log(photons_per_pixel)
        return y

    y_true_photons = get_photons(y_true)
    y_pred_photons = get_photons(y_pred)
    y_pred_photons_log = get_photons_log(y_pred)

    if include_const:
        return np.sum(y_pred_photons - y_true_photons * y_pred_photons_log
                      - np.log(factorial(y_true_photons)))
    else:
        return np.sum(y_pred_photons - y_true_photons * y_pred_photons_log)

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

recos = []
projection_obs_pois_reg_losses = []  # note 2021-01-13: corrected this variable name from projection_obs_msees (fixed syntax analysis errors and seemed obvious)
projection_ideal_obs_pois_reg_losses = []
projection_obs_mses = []
projection_ideal_obs_mses = []
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
        projection = np.asarray(ray_trafo(reco), dtype=np.float64)
        obs_float64 = np.asarray(obs, dtype=np.float64)
        ideal_obs = np.asarray(ray_trafo(gt), dtype=np.float64)

        projection_obs_pois_reg_loss = poisson_loss(
            projection, obs_float64,  # projection is first argument, because it is the prediction for the parameters, under which the likelihood of the observation ("y_true") should be evaluated
            include_const=False)
        projection_obs_pois_reg_losses.append(
            projection_obs_pois_reg_loss)

        projection_ideal_obs_pois_reg_loss = poisson_loss(
            projection, ideal_obs,  # maybe this value is not too meaningful, because projection is not expected to poisson distributed
            include_const=False)
        projection_ideal_obs_pois_reg_losses.append(
            projection_ideal_obs_pois_reg_loss)

        projection_obs_mse = np.mean(np.square(projection-obs_float64))
        projection_obs_mses.append(
            projection_obs_mse)

        projection_ideal_obs_mse = np.mean(np.square(projection-ideal_obs))
        projection_ideal_obs_mses.append(
            projection_ideal_obs_mse)

if save_data_discrepancy:
    os.makedirs(DATA_DISCREPANCY_PATH, exist_ok=True)
    # use similar style as grand-challenge.org
    data_discrepancy = {
        'case': {
            'projection_obs_pois_reg_loss': {
                str(i): m
                for i, m in enumerate(projection_obs_pois_reg_losses)},
            'projection_ideal_obs_pois_reg_loss': {
                str(i): m
                for i, m in enumerate(projection_ideal_obs_pois_reg_losses)},
            'projection_obs_mse': {
                str(i): m
                for i, m in enumerate(projection_obs_mses)},
            'projection_ideal_obs_mse': {
                str(i): m
                for i, m in enumerate(projection_ideal_obs_mses)},
            },
        'aggregates': {
            'projection_obs_pois_reg_loss_max': np.max(
                projection_obs_pois_reg_losses),
            'projection_obs_pois_reg_loss_min': np.min(
                projection_obs_pois_reg_losses),
            'projection_obs_pois_reg_loss_mean': np.mean(
                projection_obs_pois_reg_losses),
            'projection_obs_pois_reg_loss_std': np.std(
                projection_obs_pois_reg_losses),
            'projection_ideal_obs_pois_reg_loss_max': np.max(
                projection_ideal_obs_pois_reg_losses),
            'projection_ideal_obs_pois_reg_loss_min': np.min(
                projection_ideal_obs_pois_reg_losses),
            'projection_ideal_obs_pois_reg_loss_mean': np.mean(
                projection_ideal_obs_pois_reg_losses),
            'projection_ideal_obs_pois_reg_loss_std': np.std(
                projection_ideal_obs_pois_reg_losses),
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
        DATA_DISCREPANCY_PATH, '{}_data_discrepancy.json'.format(name))
    with open(filename, 'w') as f:
        json.dump(data_discrepancy, f, indent=2)
print()
print('mean poisson_loss(projection, obs)',
      np.mean(projection_obs_pois_reg_losses))
print('mean poisson_loss(projection, ideal_obs)',
      np.mean(projection_ideal_obs_pois_reg_losses))
print('mean (obs - projection)**2',
      np.mean(projection_obs_mses))
print('mean (projection - ideal_obs)**2',
      np.mean(projection_ideal_obs_mses))
