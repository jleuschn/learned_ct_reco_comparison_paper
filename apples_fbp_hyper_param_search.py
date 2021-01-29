# -*- coding: utf-8 -*-
import os
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dival.reconstructors.learnedpd_reconstructor import LearnedPDReconstructor
from dival.measure import PSNR, SSIM
from dival.evaluation import TaskTable
from dival.reconstructors.odl_reconstructors import FBPReconstructor
from dival.util.plot import plot_images
from util.apples_dataset import get_apples_dataset
from util.callback_apply_after import CallbackApplyAfter

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'
TIMESTR = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

NOISE_SETTING_DEFAULT = 'scattering'
NUM_ANGLES_DEFAULT = 50
NUM_IMAGES_DEFAULT = 100

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)
parser.add_argument('--num_images', type=int, default=NUM_IMAGES_DEFAULT)

options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
name = 'apples_fbp_{}_{:02d}_hyper_param_search'.format(
    noise_setting, num_angles)
num_images = options.num_images

dataset = get_apples_dataset(num_angles=num_angles,
                             noise_setting=noise_setting,
                             impl=IMPL)

ray_trafo = dataset.ray_trafo

pre_processor = ((lambda obs: obs / 400) if noise_setting == 'scattering'
                 else None)
reconstructor = FBPReconstructor(ray_trafo, pre_processor=pre_processor)
hyper_param_choices = {
    'filter_type': ['Cosine', 'Hann'],
    'frequency_scaling': [*np.linspace(0.06, 0.14, 9)],
    }

test_data = dataset.get_data_pairs_per_index('validation',
                                             list(range(100, 100+num_images)))
test_data.name = 'validation part[100:{:d}]'.format(100+num_images)
task_table = TaskTable('apples_fbp_hyper_param_search')
task_table.append(reconstructor, test_data, measures=[PSNR, SSIM],
                  hyper_param_choices=hyper_param_choices)
task_table.run(save_reconstructions=False)
with open(
        os.path.join(
            RESULTS_PATH, '{}_{}.txt'.format(name, TIMESTR)),
        'w') as f:
    f.write(task_table.results.to_string(show_columns='misc'))
    print(task_table.results.to_string(show_columns='misc'))

best_loss = np.inf
best_hyper_params = None
for row in task_table.results.results.iloc:
    loss = (-np.mean(row.measure_values['psnr'])
            -40*np.mean(row.measure_values['ssim']))
    if loss < best_loss and row.misc['hp_choice']['filter_type'] == 'Cosine':
        best_hyper_params = row.misc['hp_choice']
        best_loss = loss
print('best loss {:f} for hyper params\n{}'.format(best_loss, best_hyper_params))

best_loss = np.inf
best_hyper_params = None
for row in task_table.results.results.iloc:
    loss = (-np.mean(row.measure_values['psnr'])
            -40*np.mean(row.measure_values['ssim']))
    if loss < best_loss and row.misc['hp_choice']['filter_type'] == 'Hann':
        best_hyper_params = row.misc['hp_choice']
        best_loss = loss
print('best loss {:f} for hyper params\n{}'.format(best_loss, best_hyper_params))

# with tqdm(islice(dataset.generator('validation'), num_images),
#                     desc="eval '{}'".format(name),
#                     total=min(num_images, dataset.get_len('validation'))) as p:
#     for obs, gt in p:
#         psnr_iterates = []
#         ssim_iterates = []
#         def plot_result_after_iters(result, iters):
#             _, ax = plot_images([result, gt])
#             ax[0].set_xlabel('PSNR: {:.2f}, SSIM: {:.3f}'.format(
#                 PSNR(result, gt), SSIM(result, gt)))
#             plt.show()
#         def append_measures(result):
#             psnr_iterates.append(PSNR(result, gt))
#             ssim_iterates.append(SSIM(result, gt))
#         callback = (
#             CallbackApplyAfter(
#                 plot_result_after_iters,
#                 range(0, reconstructor.iterations, 1000)) &
#             CallbackApplyAfter(
#                 append_measures,
#                 range(0, reconstructor.iterations, 1)))
#         reco = reconstructor.reconstruct(obs, callback=callback)
#         plt.figure()
#         plt.subplot(2, 1, 1)
#         plt.plot(psnr_iterates)
#         plt.subplot(2, 1, 2)
#         plt.plot(ssim_iterates)
#         psnr = PSNR(reco, gt)
#         ssim = SSIM(reco, gt)
#         psnrs.append(psnr)
#         ssims.append(ssim)
#         print(np.mean(reco)/np.mean(gt))
#         p.set_postfix({'running psnr': np.mean(psnrs),
#                        'running ssim': np.mean(ssims)})
# print()
# print('mean psnr:', np.mean(psnrs))
# print('mean ssim:', np.mean(ssims))
# plot_images([reco, gt], fig_size=(10, 4))
