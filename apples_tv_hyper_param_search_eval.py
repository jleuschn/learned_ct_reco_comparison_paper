# -*- coding: utf-8 -*-
import os
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'

NOISE_SETTING_DEFAULT = 'scattering'
NUM_ANGLES_DEFAULT = 50

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)

options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
name = 'apples_tv_{}_{:02d}_hyper_param_search'.format(
    noise_setting, num_angles)

timestr = sorted([f[len(name)+1:len(name)+20] for f in os.listdir(RESULTS_PATH)
                  if f.startswith(name + '_') and f.endswith('.txt')])[-1]

with open(os.path.join(RESULTS_PATH, '{}_{}.txt'.format(name, timestr))) as f:
    text = f.read()

groups = re.findall("psnr: ([\.e\-\d]+), "
                    "ssim: ([\.e\-\d]+).*"
                    "'lr': ([\.e\-\d]+), "
                    "'gamma': ([\.e\-\d]+), "
                    "'iterations': (\d+)", text)
psnr = np.array([float(g[0]) for g in groups])
ssim = np.array([float(g[1]) for g in groups])
lr = np.array([float(g[2]) for g in groups])
gamma = np.array([float(g[3]) for g in groups])
iterations = np.array([float(g[4]) for g in groups])

plt.figure(figsize=(12, 8))
ax_psnr = plt.subplot(2, 1, 1, label='psnr')
ax_ssim = plt.subplot(2, 1, 2, label='ssim')
lines_psnr = []
labels = []
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.', ':']
for i, l in enumerate(np.unique(lr)):
    linestyle = linestyles[i % len(linestyles)]
    for j, g in enumerate(np.unique(gamma)):
        color = colors[j % len(colors)]
        mask = np.logical_and(lr==l, gamma==g)
        line_psnr, = ax_psnr.plot(iterations[mask], psnr[mask],
                                  color=color,
                                  linestyle=linestyle)
        lines_psnr.append(line_psnr)
        labels.append('lr={:g}, gamma={:g}'.format(l, g))
        ax_ssim.plot(iterations[mask], ssim[mask],
                     color=color,
                     linestyle=linestyle)
plt.figlegend(lines_psnr, labels, loc='right', bbox_to_anchor=(1.125, 0.5))

loss = -psnr - 40*ssim
i = np.argmin(loss)
print('min(-psnr - 40*ssim)={:f} at:'.format(loss[i]))
print('lr', lr[i])
print('gamma', gamma[i])
print('iterations', iterations[i])
