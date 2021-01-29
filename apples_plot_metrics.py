# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.lines as mlines
import matplotlib.collections as mcollections
import seaborn as sns
import pandas as pd

IMPL = 'astra_cuda'
METRICS_PATH = '../learned_ct_reco_comparison_paper_results/metrics'

FIG_PATH = 'figures'
os.makedirs(FIG_PATH, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--methods', type=str, nargs='+', default=None)
parser.add_argument('--exclude_methods', type=str, nargs='+', default=[])
parser.add_argument('--save_formats', type=str, nargs='+',
                    default=['png', 'pdf'])
                    # default=[])
parser.add_argument('--plot_types', type=str, nargs='+',
                    default=[
                        'num_angles_on_x',
                        'noise_setting_on_x',
                        # 'noise_setting_and_num_angles_on_x_no_method_names',
                        'noise_setting_and_num_angles_on_x'
                    ])

options = parser.parse_args()

methods = options.methods
if methods is None:
    methods = ['learnedpd', 'fbpistaunet', 'fbpunet', 'fbpmsdnet', 'cinn', 'ictnet', 'tv', 'cgls', 'fbp']
exclude_methods = options.exclude_methods
save_formats = options.save_formats
plot_types = options.plot_types

method_list = [m for m in methods if m not in exclude_methods]
num_angles_list = [50, 10, 5, 2]
noise_setting_list = ['noisefree', 'gaussian_noise', 'scattering']
noise_setting_name_dict = {
    'noisefree': 'Noise-free',
    'gaussian_noise': 'Gaussian noise',
    'scattering': 'Scattering'}
noise_setting_name_list = [
    noise_setting_name_dict[noise_setting]
    for noise_setting in noise_setting_list]
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

def get_metrics(noise_setting, num_angles, method):
    name = 'apples_{}_{:02d}_{}'.format(noise_setting, num_angles, method)
    try:
        with open(os.path.join(METRICS_PATH, '{}_metrics.json'.format(name)),
                  'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    return metrics

aggregate_keys = [
    'psnr_mean', 'psnr_std', 'psnr_min', 'psnr_max',
    'ssim_mean', 'ssim_std', 'ssim_min', 'ssim_max',
    'psnr_data_range_fixed_mean', 'psnr_data_range_fixed_std',
    'psnr_data_range_fixed_min', 'psnr_data_range_fixed_max',
    'ssim_data_range_fixed_mean', 'ssim_data_range_fixed_std',
    'ssim_data_range_fixed_min', 'ssim_data_range_fixed_max']

records = [
    {'noise_setting_name': noise_setting_name_dict[n],
     'num_angles': a,
     'method_name': method_name_dict[m],
     **get_metrics(n, a, m).get('aggregates',
                                {k: np.nan for k in aggregate_keys})
    }
    for n in noise_setting_list for a in num_angles_list for m in method_list]

df = pd.DataFrame.from_records(records)

if 'num_angles_on_x' in plot_types:
    for noise_setting in noise_setting_list:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        for method in method_list:
            metrics_per_angle = [
                get_metrics(noise_setting, num_angles, method)
                for num_angles in num_angles_list]
            psnr_per_angle = [m.get('aggregates', {}).get('psnr_mean', np.nan)
                              for m in metrics_per_angle]
            ssim_per_angle = [m.get('aggregates', {}).get('ssim_mean', np.nan)
                              for m in metrics_per_angle]
            fig.suptitle(noise_setting_name_dict[noise_setting])
            ax[0].plot(num_angles_list, psnr_per_angle, 'D--',
                       label=method_name_dict[method])
            ax[1].plot(num_angles_list, ssim_per_angle, 'D--',
                       label=method_name_dict[method])
        ax[0].set_xscale('log')
        ax[0].set_xticks(num_angles_list)
        ax[0].get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax[0].invert_xaxis()
        ax[0].set_xlabel('Number of angles')
        ax[0].set_ylabel('PSNR (dB)')
        ax[0].grid()
        ax[1].set_xscale('log')
        ax[1].set_xticks(num_angles_list)
        ax[1].get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax[1].invert_xaxis()
        ax[1].set_xlabel('Number of angles')
        ax[1].set_ylabel('SSIM')
        ax[1].grid()
        ax[1].legend(loc='lower left')
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH, 'apples_metrics_{}.pdf'.format(noise_setting))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH, 'apples_metrics_{}.png'.format(noise_setting))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'noise_setting_on_x' in plot_types:
    for num_angles in num_angles_list:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        fig.subplots_adjust(wspace=0.3)
        method_name_list = [method_name_dict[m] for m in method_list]
        x, hue = np.meshgrid(noise_setting_name_list, method_name_list)
        x, hue = x.ravel(), hue.ravel()
        metrics_per_setting_per_method = [
            get_metrics(noise_setting, num_angles, method)
            for method in method_list for noise_setting in noise_setting_list]
        y_psnr = [m.get('aggregates', {}).get('psnr_mean', np.nan)
                  for m in metrics_per_setting_per_method]
        y_ssim = [m.get('aggregates', {}).get('ssim_mean', np.nan)
                  for m in metrics_per_setting_per_method]
        fig.suptitle('{:d} angles'.format(num_angles))
        sns.swarmplot(x, y_psnr, hue, ax=ax[0])
        ax[0].set_ylabel('PSNR (dB)')
        ax[0].get_legend().remove()
        sns.swarmplot(x, y_ssim, hue, ax=ax[1])
        ax[1].set_ylabel('SSIM')
        ax[1].get_legend()._loc = 6  # center left
        ax[1].get_legend().set_bbox_to_anchor((1.05, 0.5))
        ax[1].get_legend().set_title('Method')
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH, 'apples_metrics_{:02d}_angles.pdf'.format(
                    num_angles))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH, 'apples_metrics_{:02d}_angles.png'.format(
                    num_angles))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
# if 'noise_setting_and_num_angles_on_x_no_method_names' in plot_types:
#     # note: bad order of noise settings swarms due to invert_xaxis + dodge
#     fig, ax = plt.subplots(1, 2, figsize=(15, 3.5))
#     sns.swarmplot(x='num_angles', y='psnr_mean', hue='noise_setting_name', data=df, dodge=True, ax=ax[0])
#     ax[0].invert_xaxis()
#     ax[0].set_ylabel('PSNR (dB)')
#     ax[0].get_legend().remove()
#     sns.swarmplot(x='num_angles', y='ssim_mean', hue='noise_setting_name', data=df, dodge=True, ax=ax[1])
#     ax[1].invert_xaxis()
#     ax[1].set_ylabel('SSIM')
#     ax[1].get_legend().set_title('')
#     ax[1].get_legend()._loc = 6  # center left
#     ax[1].get_legend().set_bbox_to_anchor((1.05, 0.5))
#     if 'pdf' in save_formats:
#         filename = os.path.join(
#             FIG_PATH, 'apples_metrics_all_in_one_no_method_names.pdf')
#         fig.savefig(filename, bbox_inches='tight')
#     if 'png' in save_formats:
#         filename_png = os.path.join(
#             FIG_PATH, 'apples_metrics_all_in_one_no_method_names.png')
#         fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'noise_setting_and_num_angles_on_x' in plot_types:
    fig, ax = plt.subplots(1, 2, figsize=(15, 3.5))
    x_offsets = np.linspace(-.25, .25, len(noise_setting_list))
    markers = ['P', 'o', 'X']
    for i, n in enumerate(noise_setting_list):
        df_noise_setting = df[df['noise_setting_name']==noise_setting_name_dict[n]]
        children_old = ax[0].get_children()
        sns.swarmplot(x='num_angles', y='psnr_mean', hue='method_name', data=df_noise_setting, dodge=False, marker=markers[i], ax=ax[0])
        for c in ax[0].get_children():
            if (isinstance(c, mcollections.PathCollection)
                    and c not in children_old):
                c.set_offsets(c.get_offsets() - [x_offsets[i], 0])
        children_old = ax[1].get_children()
        sns.swarmplot(x='num_angles', y='ssim_mean', hue='method_name', data=df_noise_setting, dodge=False, marker=markers[i], ax=ax[1])
        for c in ax[1].get_children():
            if (isinstance(c, mcollections.PathCollection)
                    and c not in children_old):
                c.set_offsets(c.get_offsets() - [x_offsets[i], 0])
        if i == 0:
            handles, labels = ax[1].get_legend_handles_labels()
    ax[0].invert_xaxis()
    ax[1].invert_xaxis()
    ax[0].set_xlabel('Number of angles')
    ax[1].set_xlabel('Number of angles')
    ax[0].set_ylabel('PSNR (dB)')
    ax[1].set_ylabel('SSIM')
    ax[0].get_legend().remove()
    method_handles, method_labels = handles[:len(method_list)], labels[:len(method_list)]
    method_handles = [mlines.Line2D([], [], color=h.get_facecolors()[0], marker='s', linestyle='') for h in method_handles]
    method_legend = ax[1].legend(method_handles, method_labels)
    method_legend._loc = 2  # upper left
    method_legend.set_bbox_to_anchor((1.05, 0.675))
    method_legend.set_title('Method')
    ax[1].add_artist(method_legend)
    noise_setting_handles = [mlines.Line2D([], [], color='k', marker=markers[i], linestyle='') for i in range(len(noise_setting_list))]
    noise_setting_labels = noise_setting_name_list
    noise_setting_legend = ax[1].legend(noise_setting_handles, noise_setting_labels)
    noise_setting_legend._loc = 3  # lower left
    noise_setting_legend.set_bbox_to_anchor((1.05, 0.675))
    noise_setting_legend.set_title('Noise setting')
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH, 'apples_metrics_all_in_one.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH, 'apples_metrics_all_in_one.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
