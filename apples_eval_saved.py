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
from dival.util.plot import plot_images
import matplotlib.pyplot as plt
from dival.measure import PSNR, SSIM, PSNRMeasure, SSIMMeasure
from util.apples_dataset import get_apples_dataset
from util import apples_data
from util import apples_data_test

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
RECONSTRUCTIONS_PATH = os.path.join(RESULTS_PATH, 'reconstructions')
METRICS_PATH = os.path.join(RESULTS_PATH, 'metrics')
METRICS_PATH_VAL = os.path.join(RESULTS_PATH, 'metrics_val')
SUPP_MATERIAL_PATH = 'supp_material'

NOISE_SETTING_DEFAULT = 'noisefree'
NUM_ANGLES_DEFAULT = 50
METHOD_DEFAULT = 'cgls'
NUM_IMAGES_DEFAULT = 100

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)
parser.add_argument('--method', type=str, default=METHOD_DEFAULT)
parser.add_argument('--num_images', type=int, default=NUM_IMAGES_DEFAULT)
parser.add_argument('--first_image', type=int, default=0)
parser.add_argument('--secret_test_set', action='store_true')
parser.add_argument('--save_metrics', action='store_true')
parser.add_argument('--save_figures_for_first', type=int, default=0)

options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
method = options.method  # 'learnedpd', 'fbpmsdnet', 'cinn'
name = 'apples_{}_{:02d}_{}'.format(noise_setting, num_angles, method)
num_images = options.num_images
first_image = options.first_image
secret_test_set = options.secret_test_set
save_metrics = options.save_metrics
save_figures_for_first = options.save_figures_for_first

dataset = get_apples_dataset(num_angles=num_angles,
                             noise_setting=noise_setting,
                             impl=IMPL)

ray_trafo = dataset.ray_trafo

if secret_test_set:
    with open(os.path.join(
            SUPP_MATERIAL_PATH, 'apples', 'test_samples.json'), 'r') as f:
        test_indices = json.load(f)['test_indices']
    ground_truth = [
        apples_data_test.get_ground_truth(idx)
        for idx in test_indices[first_image:first_image+num_images]]
else:
    ground_truth = [
        dataset.get_sample(idx, part='validation', out=(False, True))[1]
        for idx in range(first_image, first_image+num_images)]
FIG_PATH = 'figures'
os.makedirs(FIG_PATH, exist_ok=True)
data_range_fixed = 0.0129353  # maximum pixel value of all training and
                              # validation ground truth images
psnr_measure_data_range_fixed = PSNRMeasure(data_range=data_range_fixed)
ssim_measure_data_range_fixed = SSIMMeasure(data_range=data_range_fixed)
recos = []
psnrs = []
ssims = []
psnrs_data_range_fixed = []
ssims_data_range_fixed = []
with tqdm(ground_truth, desc="eval '{}'".format(name)) as p:
    for i, gt in enumerate(p, start=first_image):
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
        psnr = PSNR(reco, gt)
        ssim = SSIM(reco, gt)
        # plot_images([reco, gt], fig_size=(12, 5))
        psnrs.append(psnr)
        ssims.append(ssim)
        psnr_data_range_fixed = psnr_measure_data_range_fixed(reco, gt)
        ssim_data_range_fixed = ssim_measure_data_range_fixed(reco, gt)
        psnrs_data_range_fixed.append(psnr_data_range_fixed)
        ssims_data_range_fixed.append(ssim_data_range_fixed)
        p.set_postfix({'running psnr': np.mean(psnrs),
                       'running ssim': np.mean(ssims)})
        if (i < first_image + save_figures_for_first
                or save_figures_for_first == -1):
            _, ax = plot_images([reco, gt], fig_size=(12, 5))
            ax[0].set_title('{} ({} angles, {})'.format(
                            method, num_angles, noise_setting))
            ax[1].set_title('ground truth')
            ax[0].set_xlabel('PSNR: {:.2f}, SSIM: {:.3f}'.format(psnr, ssim))
            plt.savefig(
                os.path.join(FIG_PATH, '{}_{:04d}.pdf'.format(name, i)),
                bbox_inches='tight')
if save_metrics:
    os.makedirs(
        METRICS_PATH if secret_test_set else METRICS_PATH_VAL,
        exist_ok=True)
    # use similar style as grand-challenge.org
    metrics = {
        'case': {
            'psnr': {str(i): p for i, p in enumerate(psnrs)},
            'ssim': {str(i): s for i, s in enumerate(ssims)},
            'psnr_data_range_fixed': {str(i): p for i, p in
                                      enumerate(psnrs_data_range_fixed)},
            'ssim_data_range_fixed': {str(i): s for i, s in
                                      enumerate(ssims_data_range_fixed)}
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
            'psnr_data_range_fixed_max': np.max(psnrs_data_range_fixed),
            'psnr_data_range_fixed_min': np.min(psnrs_data_range_fixed),
            'psnr_data_range_fixed_mean': np.mean(psnrs_data_range_fixed),
            'psnr_data_range_fixed_std': np.std(psnrs_data_range_fixed),
            'ssim_data_range_fixed_max': np.max(ssims_data_range_fixed),
            'ssim_data_range_fixed_min': np.min(ssims_data_range_fixed),
            'ssim_data_range_fixed_mean': np.mean(ssims_data_range_fixed),
            'ssim_data_range_fixed_std': np.std(ssims_data_range_fixed),
            }
        }
    filename = os.path.join(
        METRICS_PATH if secret_test_set else METRICS_PATH_VAL,
        '{}_metrics.json'.format(name))
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=2)
print()
print('mean PSNR:', np.mean(psnrs))
print('mean SSIM:', np.mean(ssims))
print('mean PSNR (data range fixed):', np.mean(psnrs_data_range_fixed))
print('mean SSIM (data range fixed):', np.mean(ssims_data_range_fixed))
