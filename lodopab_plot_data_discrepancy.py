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
                        'boxplot_pois_reg_loss',
                        'boxplot_mse',
                        'scatter',
                        # 'scatter_pois_reg_loss_three_ways',
                        # 'scatter_mse_three_ways',
                        'versus_performance_mean',
                        'versus_psnr_mean',
                        # 'versus_performance_mean_pois_reg_loss_three_ways',
                        # 'versus_performance_mean_mse_three_ways',
                        # 'versus_performance'
                    ])

options = parser.parse_args()

methods = options.methods
if methods is None:
    methods = ['learnedpd', 'fbpistaunet', 'fbpunet', 'fbpmsdnet', 'fbpunetpp', 'cinn', 'diptv', 'ictnet', 'tv', 'fbp']
exclude_methods = options.exclude_methods
save_formats = options.save_formats
plot_types = options.plot_types

method_list = [m for m in methods if m not in exclude_methods]
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

def get_data_discrepancy(method):
    name = 'lodopab_{}'.format(method)
    try:
        with open(os.path.join(DATA_DISCREPANCY_PATH,
                               '{}_data_discrepancy.json'.format(name)),
                  'r') as f:
            data_discrepancy = json.load(f)
    except FileNotFoundError:
        data_discrepancy = {}
    return data_discrepancy

def get_metrics(method):
    name = 'lodopab_{}'.format(method)
    try:
        with open(os.path.join(METRICS_PATH, '{}_metrics.json'.format(name)),
                  'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    return metrics

# aggregate_keys = [
#     'projection_obs_pois_reg_loss_mean', 'projection_obs_pois_reg_loss_std',
#     'projection_obs_pois_reg_loss_min', 'projection_obs_pois_reg_loss_max',
#     'projection_ideal_obs_mse_mean', 'projection_ideal_obs_mse_std',
#     'projection_ideal_obs_mse_min', 'projection_ideal_obs_mse_max']
        
# records = [
#     {'method_name': method_name_dict[m],
#      **get_data_discrepancy(m).get('aggregates',
#                                    {k: np.nan for k in aggregate_keys})
#     }
#     for m in method_list]

# df = pd.DataFrame.from_records(records)

projection_obs_pois_reg_loss_per_method = []
projection_ideal_obs_pois_reg_loss_per_method = []
projection_obs_mse_per_method = []
projection_ideal_obs_mse_per_method = []
psnr_data_range_1_per_method = []
ssim_data_range_1_per_method = []
for m in method_list:
    data_discrepancy = get_data_discrepancy(m)

    values_dict = data_discrepancy.get('case', {}).get(
        'projection_obs_pois_reg_loss', {})
    values = [values_dict[str(i)] for i in sorted(values_dict.keys())]
    projection_obs_pois_reg_loss_per_method.append(values)

    values_dict = data_discrepancy.get('case', {}).get(
        'projection_ideal_obs_pois_reg_loss', {})
    values = [values_dict[str(i)] for i in sorted(values_dict.keys())]
    projection_ideal_obs_pois_reg_loss_per_method.append(values)

    values_dict = data_discrepancy.get('case', {}).get(
        'projection_obs_mse', {})
    values = [values_dict[str(i)] for i in sorted(values_dict.keys())]
    projection_obs_mse_per_method.append(values)

    values_dict = data_discrepancy.get('case', {}).get(
        'projection_ideal_obs_mse', {})
    values = [values_dict[str(i)] for i in sorted(values_dict.keys())]
    projection_ideal_obs_mse_per_method.append(values)

    metrics = get_metrics(m)
    psnr_data_range_1_values_dict = metrics.get('case', {}).get(
        'psnr_data_range_1', {})
    psnr_data_range_1_values = [
        psnr_data_range_1_values_dict[str(i)]
        for i in sorted(psnr_data_range_1_values_dict.keys())]
    psnr_data_range_1_per_method.append(psnr_data_range_1_values)
    ssim_data_range_1_values_dict = metrics.get('case', {}).get(
        'ssim_data_range_1', {})
    ssim_data_range_1_values = [
        ssim_data_range_1_values_dict[str(i)]
        for i in sorted(ssim_data_range_1_values_dict.keys())]
    ssim_data_range_1_per_method.append(ssim_data_range_1_values)


data_discrepancy = get_data_discrepancy('ideal_obs_obs')

values_dict = data_discrepancy.get('case', {}).get(
    'ideal_obs_obs_pois_reg_loss', {})
ideal_obs_obs_pois_reg_loss = [
    values_dict[str(i)] for i in sorted(values_dict.keys())]

values_dict = data_discrepancy.get('case', {}).get(
    'ideal_obs_obs_mse', {})
ideal_obs_obs_mse = [
    values_dict[str(i)] for i in sorted(values_dict.keys())]


if 'boxplot_pois_reg_loss' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    labels = [method_name_dict[m] for m in method_list]
    xlim = (0.5, len(method_list)+0.5)
    ax.set_xlim(xlim)
    median_ideal_obs_obs_pois_reg_loss = np.median(ideal_obs_obs_pois_reg_loss)
    h = ax.hlines(
        median_ideal_obs_obs_pois_reg_loss, *xlim, linestyle='--',
        label=(
            'median of '
            '$-\ell_\mathrm{Pois}(A x^\dagger\,|\,y_\delta)$'))
    ax.boxplot(projection_obs_pois_reg_loss_per_method, labels=labels)
    ax.set_title('Poisson regression loss '
                 '$-\ell_\mathrm{Pois}(A \hat{x}\,|\,y_\delta)$')
    ax.legend(handles=[h])
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_boxplot_pois_reg_loss.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_boxplot_pois_reg_loss.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'boxplot_mse' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    labels = [method_name_dict[m] for m in method_list]
    xlim = (0.5, len(method_list)+0.5)
    ax.set_xlim(xlim)
    median_ideal_obs_obs_mse = np.median(ideal_obs_obs_mse)
    h = ax.hlines(median_ideal_obs_obs_mse, *xlim, linestyle='--',
                  label='median of MSE $(y_\delta - A x^\dagger)^2$')
    ax.set_yscale('log')
    ax.set_title('MSE $(y_\delta - A \hat{x})^2$')
    ax.boxplot(projection_obs_mse_per_method, labels=labels)
    ax.legend(handles=[h])
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_boxplot_mse.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_boxplot_mse.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'scatter' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    mean_values_per_method = [
        np.mean(v) for v in projection_obs_pois_reg_loss_per_method]
    labels = [method_name_dict[m] for m in method_list]
    ax.scatter(labels, mean_values_per_method, marker='x')
    xlim = (-0.5, len(method_list)-0.5)
    ax.set_xlim(xlim)
    ax.set_ylabel('mean Poisson regression loss '
                  '$-\ell_\mathrm{Pois}(A \hat{x}\,|\,y_\delta)$')
    mean_ideal_obs_obs_pois_reg_loss = np.mean(ideal_obs_obs_pois_reg_loss)
    h = ax.hlines(
        mean_ideal_obs_obs_pois_reg_loss, *xlim, linestyle='--', zorder=1,
        label=(
            'mean of '
            '$-\ell_\mathrm{Pois}(A x^\dagger\,|\,y_\delta)$'))
    ax.legend(handles=[h])
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_scatter.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_scatter.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'scatter_pois_reg_loss_three_ways' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    mean_projection_obs_pois_reg_loss_per_method = [
        np.mean(v) for v in projection_obs_pois_reg_loss_per_method]
    mean_projection_ideal_obs_pois_reg_loss_per_method = [
        np.mean(v) for v in projection_ideal_obs_pois_reg_loss_per_method]
    labels = [method_name_dict[m] for m in method_list]
    xlim = (-0.5, len(method_list)-0.5)
    ax.set_xlim(xlim)
    ax.scatter(labels, mean_projection_obs_pois_reg_loss_per_method,
               marker='D', label='pois_reg_loss(projection | obs)')
    ax.scatter(labels, mean_projection_ideal_obs_pois_reg_loss_per_method,
               marker='D', label='pois_reg_loss(projection | ideal_obs)')
    mean_ideal_obs_obs_pois_reg_loss = np.mean(ideal_obs_obs_pois_reg_loss)
    ax.hlines(mean_ideal_obs_obs_pois_reg_loss, *xlim,
              linestyle='--', label='pois_reg_loss(ideal_obs | obs)')
    ax.legend()
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_scatter_pois_reg_loss_three_ways.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_scatter_pois_reg_loss_three_ways.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'scatter_mse_three_ways' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))
    mean_projection_obs_mse_per_method = [
        np.mean(v) for v in projection_obs_mse_per_method]
    mean_projection_ideal_obs_mse_per_method = [
        np.mean(v) for v in projection_ideal_obs_mse_per_method]
    labels = [method_name_dict[m] for m in method_list]
    xlim = (-0.5, len(method_list)-0.5)
    ax.set_xlim(xlim)
    ax.scatter(labels, mean_projection_obs_mse_per_method,
               marker='D', label='mean (obs - projection)^2')
    ax.scatter(labels, mean_projection_ideal_obs_mse_per_method,
               marker='D', label='mean (projection - ideal_obs)^2')
    mean_ideal_obs_obs_mse = np.mean(ideal_obs_obs_mse)
    ax.hlines(mean_ideal_obs_obs_mse, *xlim,
              linestyle='--', label='mean (obs - ideal_obs)^2')
    ax.set_yscale('log')
    ax.legend(loc='center left')
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_scatter_mse_three_ways.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH, 'lodopab_data_discrepancy_scatter_mse_three_ways.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'versus_performance_mean' in plot_types:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    mean_values_per_method = [
        np.mean(v) for v in projection_obs_pois_reg_loss_per_method]
    mean_psnr_data_range_1_per_method = [
        np.mean(v) for v in psnr_data_range_1_per_method]
    mean_ssim_data_range_1_per_method = [
        np.mean(v) for v in ssim_data_range_1_per_method]
    xlim0 = (np.min(mean_psnr_data_range_1_per_method) - 0.5,
             np.max(mean_psnr_data_range_1_per_method) + 0.5)
    ax[0].set_xlim(xlim0)
    xlim1 = (np.min(mean_ssim_data_range_1_per_method) - 0.01,
             np.max(mean_ssim_data_range_1_per_method) + 0.01)
    ax[1].set_xlim(xlim1)
    mean_ideal_obs_obs_pois_reg_loss = np.mean(ideal_obs_obs_pois_reg_loss)
    ax[0].hlines(
        mean_ideal_obs_obs_pois_reg_loss, *xlim0, linestyle='--', zorder=1,
        label=(
            'mean of '
            '$-\ell_\mathrm{Pois}(A x^\dagger\,|\,y_\delta)$'))
    ax[1].hlines(
        mean_ideal_obs_obs_pois_reg_loss, *xlim1, linestyle='--', zorder=1,
        label=(
            'mean of '
            '$-\ell_\mathrm{Pois}(A x^\dagger\,|\,y_\delta)$'))
    for m, v, v_psnr, v_ssim in zip(
            method_list, mean_values_per_method,
            mean_psnr_data_range_1_per_method,
            mean_ssim_data_range_1_per_method):
        ax[0].scatter(
            v_psnr, v, label=method_name_dict[m], marker='D')
        ax[0].set_xlabel('mean PSNR-FR')
        ax[0].set_ylabel(
            'mean Poisson regression loss '
            '$-\ell_\mathrm{Pois}(A \hat{x}\,|\,y_\delta)$')
        ax[1].scatter(
            v_ssim, v, label=method_name_dict[m], marker='D')
        ax[1].set_xlabel('mean SSIM-FR')
        ax[1].set_ylabel(
            'mean Poisson regression loss '
            '$-\ell_\mathrm{Pois}(A \hat{x}\,|\,y_\delta)$')
        ax[1].legend(loc='center left')
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH,
                'lodopab_data_discrepancy_versus_performance_mean.pdf')
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH,
                'lodopab_data_discrepancy_versus_performance_mean.png')
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'versus_psnr_mean' in plot_types:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    mean_values_per_method = [
        np.mean(v) for v in projection_obs_pois_reg_loss_per_method]
    mean_psnr_data_range_1_per_method = [
        np.mean(v) for v in psnr_data_range_1_per_method]
    xlim = (np.min(mean_psnr_data_range_1_per_method) - 0.5,
            np.max(mean_psnr_data_range_1_per_method) + 0.5)
    ax.set_xlim(xlim)
    mean_ideal_obs_obs_pois_reg_loss = np.mean(ideal_obs_obs_pois_reg_loss)
    ideal_obs_handle = ax.hlines(
        mean_ideal_obs_obs_pois_reg_loss, *xlim, linestyle='--', color='k',
        zorder=1, label='Ground truth')
    handles = []
    for m, v, v_psnr in zip(
            method_list, mean_values_per_method,
            mean_psnr_data_range_1_per_method):
        h = ax.scatter(
            v_psnr, v, label=method_name_dict[m], marker='D')
        handles.append(h)
        ax.set_xlabel('mean PSNR-FR')
        ax.set_ylabel('mean Poisson regression loss $-\ell_\mathrm{Pois}$')
    handles.append(ideal_obs_handle)
    ax.legend(handles=handles, loc='center left')
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_versus_psnr_mean.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_versus_psnr_mean.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'versus_performance_mean_pois_reg_loss_three_ways' in plot_types:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    mean_projection_obs_pois_reg_loss_per_method = [
        np.mean(v) for v in projection_obs_pois_reg_loss_per_method]
    mean_projection_ideal_obs_pois_reg_loss_per_method = [
        np.mean(v) for v in projection_ideal_obs_pois_reg_loss_per_method]
    mean_psnr_data_range_1_per_method = [
        np.mean(v) for v in psnr_data_range_1_per_method]
    mean_ssim_data_range_1_per_method = [
        np.mean(v) for v in ssim_data_range_1_per_method]
    xlim0 = (np.min(mean_psnr_data_range_1_per_method) - 0.3,
             np.max(mean_psnr_data_range_1_per_method) + 0.3)
    ax[0].set_xlim(xlim0)
    xlim1 = (np.min(mean_ssim_data_range_1_per_method) - 0.005,
             np.max(mean_ssim_data_range_1_per_method) + 0.005)
    ax[1].set_xlim(xlim1)
    mean_ideal_obs_obs_pois_reg_loss = np.mean(ideal_obs_obs_pois_reg_loss)
    ax[0].hlines(mean_ideal_obs_obs_pois_reg_loss, *xlim0,
                 linestyle='--', label='pois_reg_loss(ideal_obs | obs)')
    ax[1].hlines(mean_ideal_obs_obs_pois_reg_loss, *xlim1,
                 linestyle='--', label='pois_reg_loss(ideal_obs | obs)')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for m, v1, v2, v_psnr, v_ssim, c in zip(
            method_list,
            mean_projection_obs_pois_reg_loss_per_method,
            mean_projection_ideal_obs_pois_reg_loss_per_method,
            mean_psnr_data_range_1_per_method,
            mean_ssim_data_range_1_per_method,
            colors):
        ax[0].scatter(
            v_psnr, v1,
            label='pois_reg_loss(projection | obs), {}'.format(
                method_name_dict[m]),
            color=c, marker='D')
        ax[0].scatter(
            v_psnr, v2,
            label='pois_reg_loss(projection | ideal_obs), {}'.format(
                method_name_dict[m]),
            color=c, marker='X')
        ax[1].scatter(
            v_ssim, v1,
            label='pois_reg_loss(projection | obs), {}'.format(
                method_name_dict[m]),
            color=c, marker='D')
        ax[1].scatter(
            v_ssim, v2,
            label='pois_reg_loss(projection | ideal_obs), {}'.format(
                method_name_dict[m]),
            color=c, marker='X')
    ax[0].set_xlabel('mean PSNR-FR')
    ax[0].set_ylabel('mean poisson regression loss')
    ax[1].set_xlabel('mean SSIM-FR')
    ax[1].set_ylabel('mean poisson regression loss')
    ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_versus_performance_mean_pois_reg_loss_three_ways.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_versus_performance_mean_pois_reg_loss_three_ways.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'versus_performance_mean_mse_three_ways' in plot_types:
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    mean_projection_obs_mse_per_method = [
        np.mean(v) for v in projection_obs_mse_per_method]
    mean_projection_ideal_obs_mse_per_method = [
        np.mean(v) for v in projection_ideal_obs_mse_per_method]
    mean_psnr_data_range_1_per_method = [
        np.mean(v) for v in psnr_data_range_1_per_method]
    mean_ssim_data_range_1_per_method = [
        np.mean(v) for v in ssim_data_range_1_per_method]
    xlim0 = (np.min(mean_psnr_data_range_1_per_method) - 0.3,
             np.max(mean_psnr_data_range_1_per_method) + 0.3)
    ax[0].set_xlim(xlim0)
    xlim1 = (np.min(mean_ssim_data_range_1_per_method) - 0.005,
             np.max(mean_ssim_data_range_1_per_method) + 0.005)
    ax[1].set_xlim(xlim1)
    mean_ideal_obs_obs_mse = np.mean(ideal_obs_obs_mse)
    ax[0].hlines(mean_ideal_obs_obs_mse, *xlim0,
                 linestyle='--', label='mean (obs - ideal_obs)^2')
    ax[1].hlines(mean_ideal_obs_obs_mse, *xlim1,
                 linestyle='--', label='mean (obs - ideal_obs)^2')
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for m, v1, v2, v_psnr, v_ssim, c in zip(
            method_list,
            mean_projection_obs_mse_per_method,
            mean_projection_ideal_obs_mse_per_method,
            mean_psnr_data_range_1_per_method,
            mean_ssim_data_range_1_per_method,
            colors):
        ax[0].scatter(
            v_psnr, v1,
            label='mean (obs - projection)^2, {}'.format(
                method_name_dict[m]),
            color=c, marker='D')
        ax[0].scatter(
            v_psnr, v2,
            label='mean (projection - ideal_obs)^2, {}'.format(
                method_name_dict[m]),
            color=c, marker='X')
        ax[1].scatter(
            v_ssim, v1,
            label='mean (obs - projection)^2, {}'.format(
                method_name_dict[m]),
            color=c, marker='D')
        ax[1].scatter(
            v_ssim, v2,
            label='mean (projection - ideal_obs)^2, {}'.format(
                method_name_dict[m]),
            color=c, marker='X')
    ax[0].set_xlabel('mean PSNR-FR')
    ax[0].set_ylabel('mse')
    ax[0].set_yscale('log')
    ax[1].set_xlabel('mean SSIM-FR')
    ax[1].set_ylabel('mse')
    ax[1].set_yscale('log')
    ax[1].legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    if 'pdf' in save_formats:
        filename = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_versus_performance_mean_mse_three_ways.pdf')
        fig.savefig(filename, bbox_inches='tight')
    if 'png' in save_formats:
        filename_png = os.path.join(
            FIG_PATH,
            'lodopab_data_discrepancy_versus_performance_mean_mse_three_ways.png')
        fig.savefig(filename_png, bbox_inches='tight', dpi=300)
if 'versus_performance' in plot_types:
    for m, v, v_psnr, v_ssim in zip(
            method_list, projection_obs_pois_reg_loss_per_method,
            psnr_data_range_1_per_method,
            ssim_data_range_1_per_method):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(method_name_dict[m])
        ax[0].scatter(v_psnr, v, marker='x', s=1)
        ax[0].set_xlabel('mean PSNR-FR')
        ax[0].set_ylabel('mean poisson regression loss')
        ax[1].scatter(v_ssim, v, marker='x', s=1)
        ax[1].set_xlabel('mean SSIM-FR')
        ax[1].set_ylabel('mean poisson regression loss')
        if 'pdf' in save_formats:
            filename = os.path.join(
                FIG_PATH,
                'lodopab_data_discrepancy_versus_performance_{}.pdf'.format(m))
            fig.savefig(filename, bbox_inches='tight')
        if 'png' in save_formats:
            filename_png = os.path.join(
                FIG_PATH,
                'lodopab_data_discrepancy_versus_performance_{}.png'.format(m))
            fig.savefig(filename_png, bbox_inches='tight', dpi=300)
