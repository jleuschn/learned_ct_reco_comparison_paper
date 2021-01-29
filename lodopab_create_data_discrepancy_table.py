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

TAB_PATH = 'tables'
os.makedirs(TAB_PATH, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--methods', type=str, nargs='+', default=None)
parser.add_argument('--exclude_methods', type=str, nargs='+', default=[])
parser.add_argument('--save_formats', type=str, nargs='+',
                    # default=['png', 'pdf'])
                    default=[])
parser.add_argument('--plot_types', type=str, nargs='+',
                    default=[
                        'boxplot_pois_reg_loss',
                        'boxplot_mse',
                        'scatter',
                        # 'scatter_pois_reg_loss_three_ways',
                        # 'scatter_mse_three_ways',
                        'versus_performance_mean',
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

aggregate_keys = [
    'projection_obs_pois_reg_loss_mean', 'projection_obs_pois_reg_loss_std',
    'projection_obs_pois_reg_loss_min', 'projection_obs_pois_reg_loss_max']

records = [
    {'method_name': method_name_dict[m],
     'data_discrepancy': get_data_discrepancy(m).get(
         'aggregates', {k: np.nan for k in aggregate_keys})
    }
    for m in method_list]

ideal_obs_obs_data_discrepancy = get_data_discrepancy('ideal_obs_obs').get(
    'aggregates', {k: np.nan for k in aggregate_keys})

df = pd.DataFrame.from_records(records)
header = ['Method', '$-\ell_\mathrm{Pois}(A \hat{x}\,|\,y_\delta)/\\num{e9}$']

def data_discrepancy_formatter_omit_e9(x):
    return '$\\num{{{:.6f}}} \\pm \\num{{{:.6f}}}$'.format(
        x['projection_obs_pois_reg_loss_mean'] * 1e-9,
        x['projection_obs_pois_reg_loss_std'] * 1e-9)

data_discrepancy_table_filename = os.path.join(
            TAB_PATH, 'lodopab_data_discrepancy_table.tex')
with open(data_discrepancy_table_filename, 'wt') as f:
    df.to_latex(
        buf=f, formatters=[None, data_discrepancy_formatter_omit_e9],
        header=header, index=False, column_format='lr', escape=False)
with open(data_discrepancy_table_filename, 'rt') as f:
    t = f.read()
    print(t)

print('ideal_obs snippet to paste before \\bottomrule:')
print('\midrule\n\midrule\n'
      '  & $-\ell_\mathrm{{Pois}}(A x^\dagger\,|\,y_\delta)/\\num{{e9}}$ '
      '\\\\\n\midrule\n'
      ' Ground truth & $\\num{{{:.6f}}} \\pm \\num{{{:.6f}}}$ \\\\'.format(
        ideal_obs_obs_data_discrepancy[
            'ideal_obs_obs_pois_reg_loss_mean'] * 1e-9,
        ideal_obs_obs_data_discrepancy[
            'ideal_obs_obs_pois_reg_loss_std'] * 1e-9)
      )
