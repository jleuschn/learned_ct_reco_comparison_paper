# -*- coding: utf-8 -*-
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
import torch
from ista_unet.evaluate import load_ista_unet_model_best_val
from dival.util.plot import plot_images
import matplotlib.pyplot as plt
from dival.measure import PSNR, SSIM
from dival.data import DataPairs
from util.apples_dataset import get_apples_dataset
from util import apples_data
from util import apples_data_test
from odl.tomo.analytic.filtered_back_projection import fbp_op
from guid_dict import guid_dict

IMPL = 'astra_cuda'
RESULTS_PATH = '../codesprint_paper_results'
RECONSTRUCTIONS_PATH = os.path.join(RESULTS_PATH, 'reconstructions')

NOISE_SETTING_DEFAULT = 'gaussian_noise'
NUM_ANGLES_DEFAULT = 50
NUM_IMAGES_DEFAULT = 100

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)
parser.add_argument('--num_images', type=int, default=NUM_IMAGES_DEFAULT)
parser.add_argument('--first_image', type=int, default=0)
parser.add_argument('--secret_test_set', action='store_true')
parser.add_argument('--save_reconstructions', action='store_true')
parser.add_argument('--save_figures_for_first', type=int, default=0)

options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
name = 'apples_{}_{:02d}_fbpistaunet'.format(noise_setting, num_angles)
num_images = options.num_images
first_image = options.first_image
secret_test_set = options.secret_test_set
save_reconstructions = options.save_reconstructions
save_figures_for_first = options.save_figures_for_first

dataset = get_apples_dataset(num_angles=num_angles,
                             noise_setting=noise_setting,
                             impl=IMPL)

ray_trafo = dataset.ray_trafo

scale_scattering_obs = True

model, config_dict = load_ista_unet_model_best_val(
    guid=guid_dict[noise_setting][num_angles],
    dataset='fbpistaunet_{}_{:02d}'.format(noise_setting, num_angles),
    return_config_dict=True)

print(config_dict)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
# fbp filter with default parameters of dival.datasets.fbp_dataset.FBPDataset
fbp_operator = fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=1.0)

if secret_test_set:
    with open(os.path.join(
            os.path.dirname(RESULTS_PATH),
            'codesprint_paper',
            'supp_material', 'apples', 'test_samples.json'), 'r') as f:
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
FIG_PATH = os.path.join(os.path.dirname(__file__), 'figures')
recos = []
psnrs = []
ssims = []
with tqdm(test_data, desc="eval '{}'".format(name)) as p:
    for i, (obs, gt) in enumerate(p, start=first_image):
        if noise_setting == 'scattering' and scale_scattering_obs:
            obs /= 400
        obs_fbp = fbp_operator(obs)
        with torch.no_grad():
            obs_fbp_tensor = torch.from_numpy(
                np.asarray(obs_fbp)).to(device)[None, None]
            reco_tensor = model(obs_fbp_tensor).clamp(0, 1)
        reco = reco_tensor[0, 0].cpu().numpy()
        psnr = PSNR(reco, gt)
        ssim = SSIM(reco, gt)
        recos.append(reco)
        psnrs.append(psnr)
        ssims.append(ssim)
        p.set_postfix({'running psnr': np.mean(psnrs),
                        'running ssim': np.mean(ssims)})
        if save_reconstructions:
            assert H5PY_AVAILABLE, 'need h5py to save reconstructions'
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
                ('fbpistaunet' if secret_test_set else
                  'fbpistaunet_val'),
                {'noisefree': 'noisefree',
                  'scattering': 'scattering',
                  'gaussian_noise': 'gaussian'}[noise_setting],
                'ang{:d}'.format(num_angles))
            filename = (
                'recon_noisy_{}.hdf5'.format(slice_id)
                if noise_setting == 'gaussian_noise' else
                'recon_{}.hdf5'.format(slice_id))
            os.makedirs(path, exist_ok=True)
            with h5py.File(os.path.join(path, filename),
                            'w') as f:
                dset = f.create_dataset('data', dataset.get_shape()[1],
                                        dtype=np.float32)
                dset.write_direct(np.ascontiguousarray(reco))
        if (i < first_image + save_figures_for_first
                or save_figures_for_first == -1):
            _, ax = plot_images([reco, gt], fig_size=(12, 5))
            ax[0].set_title('fbpistaunet ({} angles, {})'.format(
                            num_angles, noise_setting))
            ax[1].set_title('ground truth')
            ax[0].set_xlabel('PSNR: {:.2f}, SSIM: {:.3f}'.format(psnr, ssim))
            plt.savefig(
                os.path.join(FIG_PATH, '{}_{:04d}{}.pdf'.format(
                    name, i, '' if secret_test_set else '_val')),
                bbox_inches='tight')
print()
print('mean psnr:', np.mean(psnrs))
print('mean ssim:', np.mean(ssims))
