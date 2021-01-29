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
DATA_DISCREPANCY_PATH = '../learned_ct_reco_comparison_paper_results/data_discrepancy'
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
                        'num_angles_on_x_log_y',
                        # 'noise_setting_on_x',
                        # 'noise_setting_and_num_angles_on_x_no_method_names',
                        'noise_setting_and_num_angles_on_x',
                        'versus_performance'
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

def get_data_discrepancy(noise_setting, num_angles, method):
    name = 'apples_{}_{:02d}_{}'.format(noise_setting, num_angles, method)
    try:
        with open(os.path.join(DATA_DISCREPANCY_PATH,
                               '{}_data_discrepancy.json'.format(name)),
                  'r') as f:
            data_discrepancy = json.load(f)
    except FileNotFoundError:
        data_discrepancy = {}
    return data_discrepancy

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
    'projection_obs_mse_mean', 'projection_obs_mse_std', 'projection_obs_mse_min', 'projection_obs_mse_max',
    'projection_ideal_obs_mse_mean', 'projection_ideal_obs_mse_std', 'projection_ideal_obs_mse_min', 'projection_ideal_obs_mse_max',
    'projection_obs_mse_data_range_fixed_mean', 'projection_obs_mse_data_range_fixed_std',
    'projection_obs_mse_data_range_fixed_min', 'projection_obs_mse_data_range_fixed_max',
    'projection_ideal_obs_mse_data_range_fixed_mean', 'projection_ideal_obs_mse_data_range_fixed_std',
    'projection_ideal_obs_mse_data_range_fixed_min', 'projection_ideal_obs_mse_data_range_fixed_max']

records = [
    {'noise_setting_name': noise_setting_name_dict[n],
     'num_angles': a,
     'method_name': method_name_dict[m],
     **get_data_discrepancy(n, a, m).get('aggregates',
                                         {k: np.nan for k in aggregate_keys})
    }
    for n in noise_setting_list for a in num_angles_list for m in method_list]

df = pd.DataFrame.from_records(records)

data_discrepancy_values_dict = {}
psnr_data_range_fixed_values_dict = {}
ssim_data_range_fixed_values_dict = {}
for m in method_list:
    data_discrepancy_values_dict.setdefault(m, {})
    psnr_data_range_fixed_values_dict.setdefault(m, {})
    ssim_data_range_fixed_values_dict.setdefault(m, {})
    for n in noise_setting_list:
        data_discrepancy_values_dict[m].setdefault(n, {})
        psnr_data_range_fixed_values_dict[m].setdefault(n, {})
        ssim_data_range_fixed_values_dict[m].setdefault(n, {})
        for a in num_angles_list:
            d = get_data_discrepancy(n, a, m).get('case', {}).get(
                'projection_obs_mse', {})
            data_discrepancy_values_dict[m][n][a] = [
                d[str(i)] for i in sorted(d.keys())]
            d = get_metrics(n, a, m).get('case', {}).get(
                'psnr_data_range_fixed', {})
            psnr_data_range_fixed_values_dict[m][n][a] = [
                d[str(i)] for i in sorted(d.keys())]
            d = get_metrics(n, a, m).get('case', {}).get(
                'ssim_data_range_fixed', {})
            ssim_data_range_fixed_values_dict[m][n][a] = [
                d[str(i)] for i in sorted(d.keys())]

if 'num_angles_on_x' in plot_types:
    for noise_setting in noise_setting_list:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
        for method in method_list:
            data_discrepancy_per_angle = [
                get_data_discrepancy(noise_setting, num_angles, method)
                for num_angles in num_angles_list]
            projection_obs_mses_per_angle = [
                m.get('aggregates', {}).get('projection_obs_mse_mean', np.nan)
                for m in data_discrepancy_per_angle]
            ax.plot(num_angles_list, projection_obs_mses_per_angle, 'D--',
                    label=method_name_dict[method])
        if noise_setting != 'noisefree':  # should be zero (in practice close)
            ideal_obs_obs_data_discrepancy_per_angle = [
                get_data_discrepancy(noise_setting, num_angles, 'ideal_obs_obs')
                for num_angles in num_angles_list]
            ideal_obs_obs_mses_per_angle = [
                m.get('aggregates', {}).get('ideal_obs_obs_mse_mean', np.nan)
                for m in ideal_obs_obs_data_discrepancy_per_angle]
            ax.plot(num_angles_list, ideal_obs_obs_mses_per_angle, '-.',
                    color='k', label='Ground truth', zorder=1)
        ax.set_title(noise_setting_name_dict[noise_setting])
        ax.set_xscale('log')
        ax.set_xticks(num_angles_list)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax.invert_xaxis()
        ax.set_xlabel('Number of angles')
        ax.set_ylabel('MSE to observation')
        ax.grid()
        loc = None
        bbox_to_anchor = None
        if noise_setting == 'noisefree':
            loc = 'upper left'
        elif noise_setting == 'gaussian_noise':
            loc = 'upper left'
        elif noise_setting == 'scattering':
            loc = 'center left'
            bbox_to_anchor = (0., 0.46)
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH, 'apples_data_discrepancy_{}.pdf'.format(noise_setting))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH, 'apples_data_discrepancy_{}.png'.format(noise_setting))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'num_angles_on_x_log_y' in plot_types:
    for noise_setting in noise_setting_list:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.75))
        for method in method_list:
            data_discrepancy_per_angle = [
                get_data_discrepancy(noise_setting, num_angles, method)
                for num_angles in num_angles_list]
            projection_obs_mses_per_angle = [
                m.get('aggregates', {}).get('projection_obs_mse_mean', np.nan)
                for m in data_discrepancy_per_angle]
            ax.plot(num_angles_list, projection_obs_mses_per_angle, 'D--',
                    label=method_name_dict[method])
        if noise_setting != 'noisefree':  # should be zero (in practice close)
            ideal_obs_obs_data_discrepancy_per_angle = [
                get_data_discrepancy(noise_setting, num_angles, 'ideal_obs_obs')
                for num_angles in num_angles_list]
            ideal_obs_obs_mses_per_angle = [
                m.get('aggregates', {}).get('ideal_obs_obs_mse_mean', np.nan)
                for m in ideal_obs_obs_data_discrepancy_per_angle]
            ax.plot(num_angles_list, ideal_obs_obs_mses_per_angle, '-.',
                    color='k', label='Ground truth', zorder=1)
        ax.set_title(noise_setting_name_dict[noise_setting])
        ax.set_xscale('log')
        ax.set_xticks(num_angles_list)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter())
        ax.invert_xaxis()
        ax.set_xlabel('Number of angles')
        ax.set_yscale('log')
        ax.set_ylabel('MSE to observation')
        ax.grid()
        loc = None
        bbox_to_anchor = None
        if noise_setting == 'noisefree':
            loc = 'lower right'
            bbox_to_anchor = (1., 0.08)
        elif noise_setting == 'gaussian_noise':
            loc = 'upper left'
            bbox_to_anchor = (0.09, 1.)
        elif noise_setting == 'scattering':
            loc = 'center right'
            bbox_to_anchor = (1., 0.38)
        ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor)
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH, 'apples_data_discrepancy_{}.pdf'.format(noise_setting))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH, 'apples_data_discrepancy_{}.png'.format(noise_setting))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'noise_setting_on_x' in plot_types:
    for num_angles in num_angles_list:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
        method_name_list = [method_name_dict[m] for m in method_list]
        x, hue = np.meshgrid(noise_setting_name_list, method_name_list)
        x, hue = x.ravel(), hue.ravel()
        data_discrepancy_per_setting_per_method = [
            get_data_discrepancy(noise_setting, num_angles, method)
            for method in method_list for noise_setting in noise_setting_list]
        y_projection_obs_mse = [
            m.get('aggregates', {}).get('projection_obs_mse_mean', np.nan)
            for m in data_discrepancy_per_setting_per_method]
        fig.suptitle('{:02d} angles'.format(num_angles))
        sns.swarmplot(x, y_projection_obs_mse, hue, ax=ax)
        ax.set_yscale('log')
        ax.set_ylabel('MSE to observation')
        ax.get_legend()._loc = 6  # center left
        ax.get_legend().set_bbox_to_anchor((1.05, 0.5))
        ax.get_legend().set_title('Method')
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH, 'apples_data_discrepancy_{:02d}_angles.pdf'.format(
                    num_angles))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH, 'apples_data_discrepancy_{:02d}_angles.png'.format(
                    num_angles))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'noise_setting_and_num_angles_on_x' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(15, 3.5))
    x_offsets = np.linspace(-.25, .25, len(noise_setting_list))
    markers = ['P', 'o', 'X']
    for i, n in enumerate(noise_setting_list):
        df_noise_setting = df[df['noise_setting_name']==noise_setting_name_dict[n]]
        children_old = ax.get_children()
        sns.swarmplot(x='num_angles', y='projection_obs_mse_mean', hue='method_name', data=df_noise_setting, dodge=False, marker=markers[i], ax=ax)
        for c in ax.get_children():
            if (isinstance(c, mcollections.PathCollection)
                    and c not in children_old):
                c.set_offsets(c.get_offsets() - [x_offsets[i], 0])
        if n != 'noisefree':  # should be zero (in practice close)
            ideal_obs_obs_data_discrepancy_per_angle = [
                get_data_discrepancy(n, num_angles, 'ideal_obs_obs')
                for num_angles in num_angles_list]
            ideal_obs_obs_mses_per_angle = [
                m.get('aggregates', {}).get('ideal_obs_obs_mse_mean', np.nan)
                for m in ideal_obs_obs_data_discrepancy_per_angle]
            ideal_obs_obs_handle = ax.plot(
                [(np.arange(len(num_angles_list))-x_offsets[i]-0.1)[::-1],
                 (np.arange(len(num_angles_list))-x_offsets[i]+0.1)[::-1]],
                [ideal_obs_obs_mses_per_angle,
                 ideal_obs_obs_mses_per_angle],
                'k--', label='Ground truth', zorder=1)[0]
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
    ax.invert_xaxis()
    ax.set_xlabel('Number of angles')
    ax.set_yscale('log')
    ax.set_ylabel('MSE to observation')
    ax.get_legend().remove()
    method_handles, method_labels = handles[:len(method_list)], labels[:len(method_list)]
    method_handles = [mlines.Line2D([], [], color=h.get_facecolors()[0], marker='s', linestyle='') for h in method_handles]
    method_handles.append(ideal_obs_obs_handle)
    method_labels.append('Ground truth')
    method_legend = ax.legend(method_handles, method_labels)
    method_legend._loc = 2  # upper left
    method_legend.set_bbox_to_anchor((1.05, 0.685))
    method_legend.set_title('Method')
    ax.add_artist(method_legend)
    noise_setting_handles = [mlines.Line2D([], [], color='k', marker=markers[i], linestyle='') for i in range(len(noise_setting_list))]
    noise_setting_labels = noise_setting_name_list
    noise_setting_legend = ax.legend(noise_setting_handles, noise_setting_labels)
    noise_setting_legend._loc = 3  # lower left
    noise_setting_legend.set_bbox_to_anchor((1.05, 0.685))
    noise_setting_legend.set_title('Noise setting')
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH, 'apples_data_discrepancy_all_in_one.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH, 'apples_data_discrepancy_all_in_one.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'versus_performance' in plot_types:
    for m in method_list:
        fig, ax = plt.subplots(
            len(noise_setting_list), 2, figsize=(15, 12))
        fig.subplots_adjust(hspace=0.3, top=0.94)
        psnr_xlim = (
            min([np.min(psnr_data_range_fixed_values_dict[m][n][a])
                 for n in noise_setting_list for a in num_angles_list]) - 1.,
            max([np.max(psnr_data_range_fixed_values_dict[m][n][a])
                 for n in noise_setting_list for a in num_angles_list]) + 1.)
        ssim_xlim = (
            min([np.min(ssim_data_range_fixed_values_dict[m][n][a])
                 for n in noise_setting_list for a in num_angles_list]) - 0.02,
            max([np.max(ssim_data_range_fixed_values_dict[m][n][a])
                 for n in noise_setting_list for a in num_angles_list]) + 0.02)
        for i, n in enumerate(noise_setting_list):
            v = data_discrepancy_values_dict[m][n]
            v_psnr = psnr_data_range_fixed_values_dict[m][n]
            v_ssim = ssim_data_range_fixed_values_dict[m][n]
            fig.suptitle('{}'.format(method_name_dict[m]))
            if n != 'noisefree':  # should be zero (in practice close)
                ideal_obs_obs_data_discrepancy_per_angle = [
                    get_data_discrepancy(n, num_angles, 'ideal_obs_obs')
                    for num_angles in num_angles_list]
                ideal_obs_obs_mses_per_angle = [
                    m.get('aggregates', {}).get('ideal_obs_obs_mse_mean',
                                                np.nan)
                    for m in ideal_obs_obs_data_discrepancy_per_angle]
                ax[i, 0].hlines(
                    ideal_obs_obs_mses_per_angle, *psnr_xlim,
                    linestyle='--', color='grey', zorder=1)
                ideal_obs_obs_handle = ax[i, 1].hlines(
                    ideal_obs_obs_mses_per_angle, *ssim_xlim,
                    linestyle='--', color='grey', zorder=1)
            for a in num_angles_list:
                ax[i, 0].scatter(v_psnr[a], v[a],
                    label='{:02d} angles'.format(a), marker='x', s=1)
            ax[i, 0].set_xlim(psnr_xlim)
            ax[i, 0].set_yscale('log')
            ax[i, 0].set_xlabel('PSNR-FR')
            ax[i, 0].set_ylabel('MSE to observation')
            ax[i, 0].set_title(noise_setting_name_dict[n])
            for a in num_angles_list:
                ax[i, 1].scatter(v_ssim[a], v[a],
                    label='{:02d} angles'.format(a), marker='x', s=1)
            ax[i, 1].set_xlim(ssim_xlim)
            ax[i, 1].set_yscale('log')
            ax[i, 1].set_xlabel('SSIM-FR')
            ax[i, 1].set_ylabel('MSE to observation')
            ax[i, 1].set_title(noise_setting_name_dict[n])
        loc = 'lower left'
        if m == 'tv':
            loc = 'upper left'
        ax[0, 1].legend(loc=loc)
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH,
                'apples_data_discrepancy_versus_performance_{}.pdf'.format(m))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH,
                'apples_data_discrepancy_versus_performance_{}.png'.format(m))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
