# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd

IMPL = 'astra_cuda'
METRICS_PATH = '../learned_ct_reco_comparison_paper_results/metrics'

TAB_PATH = 'tables'
os.makedirs(TAB_PATH, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--methods', type=str, nargs='+', default=None)
parser.add_argument('--exclude_methods', type=str, nargs='+', default=[])

options = parser.parse_args()

methods = options.methods
if methods is None:
    methods = ['learnedpd', 'fbpistaunet', 'fbpunet', 'fbpmsdnet', 'cinn', 'ictnet', 'tv', 'cgls', 'fbp']
exclude_methods = options.exclude_methods

method_list = [m for m in methods if m not in exclude_methods]
num_angles_list = [50, 10, 5, 2]
noise_setting_list = ['noisefree', 'gaussian_noise', 'scattering']
noise_setting_name_dict = {
    'noisefree': 'Noise-free',
    'gaussian_noise': 'Gaussian noise',
    'scattering': 'Scattering'}

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

metrics = {n: {a: {m: get_metrics(n, a, m).get(
                          'aggregates', {k: np.nan for k in aggregate_keys})
                   for m in method_list}
               for a in num_angles_list}
           for n in noise_setting_list}

for data_range_fixed in [False, True]:
    suffix = '_data_range_fixed' if data_range_fixed else ''
    psnr_mean_key = (
        'psnr_data_range_fixed_mean' if data_range_fixed else 'psnr_mean')
    psnr_std_key = (
        'psnr_data_range_fixed_std' if data_range_fixed else 'psnr_std')
    ssim_mean_key = (
        'ssim_data_range_fixed_mean' if data_range_fixed else 'ssim_mean')
    ssim_std_key = (
        'ssim_data_range_fixed_std' if data_range_fixed else 'ssim_std')
    
    mean_table_records = [
        {'method_name': method_name_dict[m],
         'noise_setting': n,
         **{'{:02d}_{}'.format(a, psnr_mean_key):
                metrics[n][a][m][psnr_mean_key]
            for a in num_angles_list},
         **{'{:02d}_{}'.format(a, ssim_mean_key):
                metrics[n][a][m][ssim_mean_key]
            for a in num_angles_list}}
        for m in method_list for n in noise_setting_list]
    std_table_records = [
        {'method_name': method_name_dict[m],
         'noise_setting': n,
         **{'{:02d}_{}'.format(a, psnr_std_key):
                metrics[n][a][m][psnr_std_key]
            for a in num_angles_list},
         **{'{:02d}_{}'.format(a, ssim_std_key):
                metrics[n][a][m][ssim_std_key]
            for a in num_angles_list}}
        for m in method_list for n in noise_setting_list]
    mean_table_df = pd.DataFrame.from_records(mean_table_records,
                                              index='method_name')
    std_table_df = pd.DataFrame.from_records(std_table_records,
                                             index='method_name')
    mean_table_df.index.name = None  # 'Method'
    std_table_df.index.name = None  # 'Method'
    mean_table_header = None  # ['50 angles PSNR', ..., '02 angles PSNR',
                              #  '50 angles SSIM', ..., '02 angles SSIM']
    std_table_header = None  # ['50 angles PSNR', ..., '02 angles PSNR',
                             #  '50 angles SSIM', ..., '02 angles SSIM']
    
    # mean table methods vertical
    for n in noise_setting_list:
        df_noise_setting = mean_table_df[
            mean_table_df['noise_setting']==n].drop(columns=['noise_setting'])
        mean_table_filename = os.path.join(
            TAB_PATH, 'apples_mean_table{}.tex'.format(suffix))
        def psnr_mean_formatter(a, x):
            if x == max((v[psnr_mean_key] for v in metrics[n][a].values())):
                return '{{\\cellcolor{{black!10}}}}{:.2f}'.format(x)
            else:
                return '{:.2f}'.format(x)
        def ssim_mean_formatter(a, x):
            if x == max((v[ssim_mean_key] for v in metrics[n][a].values())):
                return '{{\\cellcolor{{black!10}}}}{:.3f}'.format(x)
            else:
                return '{:.3f}'.format(x)
        formatters = (
            [lambda x, a=a:
             psnr_mean_formatter(a, x) for a in num_angles_list] +
            [lambda x, a=a:
             ssim_mean_formatter(a, x) for a in num_angles_list])
        with open(mean_table_filename, 'wt') as f:
            df_noise_setting.to_latex(
                buf=f, formatters=formatters, header=mean_table_header,
                column_format=('lr'+'r'*2*len(num_angles_list)), escape=False)
        with open(mean_table_filename, 'rt') as f:
            t = f.read()
            print(noise_setting_name_dict[n])
            print('Mean (data range fixed):' if data_range_fixed else 'Mean:')
            print(t)
    
    # std table methods vertical
    for n in noise_setting_list:
        df_noise_setting = std_table_df[
            std_table_df['noise_setting']==n].drop(columns=['noise_setting'])
        std_table_filename = os.path.join(
            TAB_PATH, 'apples_std_table{}.tex'.format(suffix))
        def psnr_std_formatter(a, x):
            return '{:.2f}'.format(x)
        def ssim_std_formatter(a, x):
            return '{:.3f}'.format(x)
        formatters = (
            [lambda x, a=a:
             psnr_std_formatter(a, x) for a in num_angles_list] +
            [lambda x, a=a:
             ssim_std_formatter(a, x) for a in num_angles_list])
        with open(std_table_filename, 'wt') as f:
            df_noise_setting.to_latex(
                buf=f, formatters=formatters, header=std_table_header,
                column_format=('lr'+'r'*2*len(num_angles_list)), escape=False)
        with open(std_table_filename, 'rt') as f:
            t = f.read()
            print(noise_setting_name_dict[n])
            print('Std (data range fixed):' if data_range_fixed else 'Std:')
            print(t)
