# -*- coding: utf-8 -*-
import os
import argparse
import json
from util.log_utils import (
    apples_convert_logs, apples_convert_logs_lightning, plot_logs)

RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'

parser = argparse.ArgumentParser()
parser.add_argument('--methods', type=str, nargs='+', default=None)
parser.add_argument('--exclude_methods', type=str, nargs='+', default=[])
parser.add_argument('--skip_conversion', action='store_true', default=False)
parser.add_argument('--save_formats', type=str, nargs='+',
                    default=['png', 'pdf'])
                    # default=[])
parser.add_argument('--plot_types', type=str, nargs='+',
                    default=[
                        'psnr',
                        'loss'
                    ])

options = parser.parse_args()

methods = options.methods
if methods is None:
    methods = ['cinn']  # ['learnedpd', 'fbpmsdnet', 'fbpunet', 'cinn']
exclude_methods = options.exclude_methods
save_formats = options.save_formats
plot_types = options.plot_types
skip_conversion = options.skip_conversion

NPZ_LOG_PATH = os.path.join(RESULTS_PATH, 'npz_logs')
os.makedirs(NPZ_LOG_PATH, exist_ok=True)
FIG_PATH = 'figures'
os.makedirs(FIG_PATH, exist_ok=True)

num_angles_list = [50, 10, 5, 2]
noise_setting_list = ['noisefree', 'gaussian_noise', 'scattering']
convert_running_to_current = False
convert_current_to_running = True

noise_setting_name_dict = {
    'noisefree': 'noise-free',
    'gaussian_noise': 'Gaussian noise',
    'scattering': 'scattering'}

method_name_dict = {'learnedpd': 'Learned Primal-Dual',
                    'fbpistaunet': 'ISTA U-Net',
                    'fbpunet': 'U-Net',
                    'fbpmsdnet': 'MS-D-CNN',
                    'fbpunetpp': 'U-Net++',
                    'cinn': 'CINN',
                    'diptv': 'DIP + TV',
                    'ictnet': 'iCTU-Net',
                    'cgls': 'CGLS',
                    'tv': 'TV',
                    'fbp': 'FBP'}

lightning_methods = ['cinn']

batch_sizes_without_hyper_params_file = {
    'cinn': (3, 3, 3, 3)
}

loss_scale_non_log_methods = ['cinn']

ylim_overrides = {
    'loss': {
        'fbpunet': {
            'noisefree': (None, 1e-3),
            'gaussian_noise': (None, 1e-3),
            'scattering': (None, 5e-3)},
        'fbpmsdnet': {
            'noisefree': (None, 2e-5),
            'gaussian_noise': (None, None),
            'scattering': (None, 1e-5)},
        'cinn': {
            'noisefree': (-5.5, -3.25),
            'gaussian_noise': (-5.5, -3.25),
            'scattering': (-5.2, -4.)}
    },
    'psnr': {
        'fbpmsdnet': {
            'noisefree': (4, None),
            'gaussian_noise': (None, None),
            'scattering': (5, None)}
    }
}

for method in methods:
    try:
        batch_sizes = batch_sizes_without_hyper_params_file[method]
    except KeyError:
        batch_sizes = (None,) * len(num_angles_list)

    for noise_setting in noise_setting_list:

        if not skip_conversion:
            for num_angles, batch_size in zip(num_angles_list, batch_sizes):
                name = 'apples_{}_{:02d}_{}'.format(
                    noise_setting, num_angles, method)
                if batch_size is None:
                    with open(os.path.join(
                            RESULTS_PATH,
                            '{}_hyper_params.json'.format(name))) as f:
                        hyper_params = json.load(f)
                    batch_size = (hyper_params['batch_size'],)
                if method in lightning_methods:
                    apples_convert_logs_lightning(
                        RESULTS_PATH, NPZ_LOG_PATH, method,
                        noise_setting=noise_setting, num_angles=num_angles,
                        convert_current_to_running=convert_current_to_running,
                        batch_sizes=batch_size)
                else:
                    apples_convert_logs(
                        RESULTS_PATH, NPZ_LOG_PATH, method,
                        noise_setting=noise_setting, num_angles=num_angles,
                        convert_running_to_current=convert_running_to_current,
                        batch_sizes=batch_size)

        for plot_type in plot_types:
            try:
                ylim = ylim_overrides[plot_type][method][noise_setting]
            except KeyError:
                ylim = None
            if plot_type == 'psnr' and method in lightning_methods:
                continue
            ax = plot_logs(
                NPZ_LOG_PATH, method, noise_setting=noise_setting,
                plot=plot_type, hyper_params_path=RESULTS_PATH,
                batch_sizes=batch_sizes, ylim=ylim,
                loss_scale_log=method not in loss_scale_non_log_methods,
                figsize=(7, 4))
            ax.set_title('{} training on {} data'.format(
                method_name_dict[method],
                noise_setting_name_dict[noise_setting]))
            if 'pdf' in save_formats:
                ax.figure.savefig(os.path.join(
                    FIG_PATH, 'apples_{}_{}_training_curve_{}.pdf'.format(
                        noise_setting, method, plot_type)),
                    bbox_inches='tight')
            if 'png' in save_formats:
                ax.figure.savefig(os.path.join(
                    FIG_PATH, 'apples_{}_{}_training_curve_{}.png'.format(
                        noise_setting, method, plot_type)),
                    bbox_inches='tight')
