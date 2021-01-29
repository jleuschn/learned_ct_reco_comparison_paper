# -*- coding: utf-8 -*-
"""
Evaluate reconstruction methods on LoDoPaB-CT.
Can save reconstructions to file.
"""
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
from dival.reconstructors.learnedpd_reconstructor import LearnedPDReconstructor
from dival.reconstructors.tvadam_ct_reconstructor import TVAdamCTReconstructor
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.reconstructors.fbpunet_reconstructor import FBPUNetReconstructor
try:
    from util.fbpmsdnet_online_reconstructor import FBPMSDNetReconstructor
    MSD_PYTORCH_AVAILABLE = True
except ImportError:
    MSD_PYTORCH_AVAILABLE = False
try:
    import pytorch_lightning as pl
    from util.cinn_reconstructor import CINNReconstructor
    CINN_AVAILABLE = True
except ImportError:
    CINN_AVAILABLE = False
# from util.fbpmsdnet_original_reconstructor import (
#     FBPMSDNetOriginalReconstructor)
from dival.util.plot import plot_images
import matplotlib.pyplot as plt
from dival.measure import PSNR, SSIM
from dival.data import DataPairs
from dival import get_standard_dataset
from util import lodopab_challenge_set
from util.lodopab_submission import save_reconstruction

lodopab_challenge_set.config['data_path'] = (
    '/localdata/lodopab_challenge_set')

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
RECONSTRUCTIONS_PATH = os.path.join(RESULTS_PATH, 'reconstructions_lodopab')
GROUND_TRUTH_PATH = '/localdata/jleuschn/low_dose_dataset_challenge'

METHOD_DEFAULT = 'fbpunet'
NUM_IMAGES_DEFAULT = -1

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default=METHOD_DEFAULT)
parser.add_argument('--num_images', type=int, default=-1)
parser.add_argument('--first_image', type=int, default=0)
parser.add_argument('--part', type=str, default='challenge')
parser.add_argument('--save_reconstructions', action='store_true')
parser.add_argument('--save_figures_for_first', type=int, default=0)

options = parser.parse_args()

method = options.method  # 'learnedpd', 'fbpmsdnet', 'cinn'
name = 'lodopab_{}'.format(method)
num_images = options.num_images
first_image = options.first_image
part = options.part
save_reconstructions = options.save_reconstructions
save_figures_for_first = options.save_figures_for_first

dataset = get_standard_dataset('lodopab', impl=IMPL)

ray_trafo = dataset.ray_trafo

if method == 'learnedpd':
    reconstructor = LearnedPDReconstructor(ray_trafo)
elif method == 'fbpunet':
    reconstructor = FBPUNetReconstructor(ray_trafo)
elif method == 'fbpmsdnet':
    assert MSD_PYTORCH_AVAILABLE
    reconstructor = FBPMSDNetReconstructor(ray_trafo)
elif method == 'cinn':
    assert CINN_AVAILABLE
    pl.seed_everything(42)
    reconstructor = CINNReconstructor(ray_trafo,
                                      max_samples_per_run=25,
                                      hyper_params={'samples_per_reco': 1000})
elif method == 'tv':
    reconstructor = TVAdamCTReconstructor(ray_trafo)
elif method == 'fbp':
    reconstructor = FBPReconstructor(ray_trafo)
else:
    raise ValueError("unknown reconstructor '{}'".format(method))

reconstructor.load_params(os.path.join(RESULTS_PATH, name))

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

FIG_PATH = 'figures'
os.makedirs(FIG_PATH, exist_ok=True)
recos = []
psnrs = []
ssims = []
if num_images == -1:
    num_images = (lodopab_challenge_set.NUM_IMAGES-first_image
                  if part == 'challenge' else
                  dataset.get_len(part)-first_image)
with tqdm(range(first_image, first_image+num_images),
          desc="eval '{}'".format(name)) as p:
    for i in p:
        obs, gt = get_data_pair(i, part)
        reco = reconstructor.reconstruct(obs)
        psnr = PSNR(reco, gt)
        ssim = SSIM(reco, gt)
        recos.append(reco)
        psnrs.append(psnr)
        ssims.append(ssim)
        p.set_postfix({'running psnr': np.mean(psnrs),
                       'running ssim': np.mean(ssims)})
        if save_reconstructions:
            path = os.path.join(
                RECONSTRUCTIONS_PATH,
                ('{}'.format(method) if part == 'challenge' else
                 '{}_{}'.format(method, part)))
            os.makedirs(path, exist_ok=True)
            save_reconstruction(path, i, reco)
        if (i < first_image + save_figures_for_first
                or save_figures_for_first == -1):
            _, ax = plot_images([reco, gt], fig_size=(12, 5))
            ax[0].set_title('{}'.format(method))
            ax[1].set_title('ground truth')
            ax[0].set_xlabel('PSNR: {:.2f}, SSIM: {:.3f}'.format(psnr, ssim))
            plt.savefig(
                os.path.join(FIG_PATH, '{}_{:04d}.pdf'.format(name, i)),
                bbox_inches='tight')
print()
print('mean psnr:', np.mean(psnrs))
print('mean ssim:', np.mean(ssims))
