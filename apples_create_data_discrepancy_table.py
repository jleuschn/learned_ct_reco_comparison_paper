# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd

IMPL = 'astra_cuda'
DATA_DISCREPANCY_PATH = '../learned_ct_reco_comparison_paper_results/data_discrepancy'

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

def get_data_discrepancy(noise_setting, num_angles, method):
    name = 'apples_{}_{:02d}_{}'.format(noise_setting, num_angles, method)
    try:
        with open(os.path.join(DATA_DISCREPANCY_PATH,
                               '{}_data_discrepancy.json'.format(name)),
                  'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}
    return metrics

aggregate_keys = [
    'projection_obs_mse_mean', 'projection_obs_mse_std',
    'projection_obs_mse_min', 'projection_obs_mse_max']

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

data_discrepancy = {n: {a: {m: get_data_discrepancy(n, a, m).get(
                       'aggregates', {k: np.nan for k in aggregate_keys})
                            for m in method_list}
                        for a in num_angles_list}
                    for n in noise_setting_list}

ideal_obs_obs_data_discrepancy = {
    n: {
        a: get_data_discrepancy(n, a, 'ideal_obs_obs').get('aggregates',
            {k: np.nan for k in aggregate_keys})
        for a in num_angles_list}
    for n in noise_setting_list}

table_records = [
    {'method_name': method_name_dict[m],
     'noise_setting': n,
     **{a: data_discrepancy[n][a][m] for a in num_angles_list}}
    for m in method_list for n in noise_setting_list]
table_df = pd.DataFrame.from_records(table_records,
                                     index='method_name')
table_df.index.name = None  # 'Method'
table_header = None  # ['50 angles', ..., '02 angles']

def formatter(x, multiply=1., digits=5, ideal_obs=False):
    cell_format = (
        '$\\num{{{{{{:.{digits:d}f}}}}}} \\pm'
        ' \\num{{{{{{:.{digits:d}f}}}}}}$'
        .format(digits=digits))
    mean_key = (
        'ideal_obs_obs_mse_mean' if ideal_obs else 'projection_obs_mse_mean')
    std_key = (
        'ideal_obs_obs_mse_std' if ideal_obs else 'projection_obs_mse_std')
    return cell_format.format(x[mean_key] * multiply, x[std_key] * multiply)

formatter_kwargs_dict = {
    'noisefree': {'multiply': 1e9, 'digits': 3},
    'gaussian_noise': {'multiply': 1e9, 'digits': 3},
    'scattering': {'multiply': 1e9, 'digits': 2}
}

# mean table methods vertical
for n in noise_setting_list:
    df_noise_setting = table_df[
        table_df['noise_setting']==n].drop(columns=['noise_setting'])
    table_filename = os.path.join(
        TAB_PATH, 'apples_data_discrepancy_table_{}.tex'.format(n))
    formatter_ = {
        'noisefree': (lambda x: formatter(x, multiply=1e9, digits=3)),
        'gaussian_noise': (lambda x: formatter(x, multiply=1e9, digits=3)),
        'scattering': (lambda x: formatter(x, multiply=1e9, digits=2))}
    formatter_ = (lambda x: formatter(x, **formatter_kwargs_dict[n]))
    formatters = [formatter_ for a in num_angles_list]
    with open(table_filename, 'wt') as f:
        df_noise_setting.to_latex(
            buf=f, formatters=formatters, header=table_header,
            column_format=('lr'+'r'*len(num_angles_list)), escape=False)
    with open(table_filename, 'rt') as f:
        t = f.read()
        print(noise_setting_name_dict[n])
        print('Data discrepancy:')
        print(t)

for n in noise_setting_list:
    formatter_ = (lambda x: formatter(x, **formatter_kwargs_dict[n],
                                      ideal_obs=True))
    ideal_obs_cells = ' & '.join([
        formatter_(ideal_obs_obs_data_discrepancy[n][a])
        for a in num_angles_list])
    print(noise_setting_name_dict[n])
    print('ideal_obs snippet to paste after:')
    print('\midrule\n'
          ' Ground truth & {} \\\\\n'.format(ideal_obs_cells))
