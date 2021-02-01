# -*- coding: utf-8 -*-
import os
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from guid_dict import guid_dict

RESULTS_PATH = '/localdata/jleuschn/ISTA-U-Net_results'

noise_setting_list = ['noisefree', 'gaussian_noise', 'scattering']

noise_setting_name_dict = {
    'noisefree': 'noise-free',
    'gaussian_noise': 'Gaussian noise',
    'scattering': 'scattering'}

do_best_val_psnr_sanity_check = True

for noise_setting in noise_setting_list:
    fig, ax = plt.subplots(figsize=(7, 4))
    for num_angles in (50, 10, 5, 2):
        guid = guid_dict[noise_setting][num_angles]
        if guid is None:
            continue
        path = os.path.join(RESULTS_PATH, 'fbpistaunet_{}_{:02d}'.format(
            noise_setting, num_angles), guid)
        filename = os.path.join(path, 'latest_validation_psnrs.txt')
        validation_psnrs = np.genfromtxt(
            filename, delimiter=',', skip_header=1)
        if os.path.isfile(os.path.join(path, 'config_dict_best_val.pickle')):
            best_val_psnr_idx = np.argmax(validation_psnrs[:, 1])
            if do_best_val_psnr_sanity_check:
                # sanity check with step stored in config_dict
                with open(os.path.join(path, 'config_dict_best_val.pickle'),
                          'rb') as handle:
                    config_dict = pickle.load(handle)
                epochs_from_lr = config_dict['scheduler'].last_epoch
                assert validation_psnrs[best_val_psnr_idx, 0] == epochs_from_lr
        else:
            best_val_psnr_idx = -1
        line = ax.plot(*validation_psnrs.T,
                       linestyle='--',
                       label='{:02d} angles'.format(num_angles))[0]
        ax.plot(*validation_psnrs[best_val_psnr_idx], 'x',
                color=line.get_color())
    # ax.set_title('fbpistaunet_{}'.format(noise_setting))
    ax.set_title('ISTA U-Net training on {} data'.format(
        noise_setting_name_dict[noise_setting]))
    ax.set_xlabel('Epochs')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(
        integer=True, steps=[1, 2, 4, 5, 10]))
    ax.set_ylabel('PSNR (dB)')
    ax.legend(loc='lower right')
    plt.savefig(
        'apples_{}_fbpistaunet_training_curve_psnr.pdf'.format(noise_setting),
        bbox_inches='tight')
    fig.savefig(
        'apples_{}_fbpistaunet_training_curve_psnr.png'.format(noise_setting),
        bbox_inches='tight')
