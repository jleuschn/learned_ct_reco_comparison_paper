# -*- coding: utf-8 -*-
"""
Train neural networks.
"""
import os
import argparse
import json
# import torch
from dival.reconstructors.learnedpd_reconstructor import LearnedPDReconstructor
from util.fbpunet_online_reconstructor import FBPUNetReconstructor
try:
    from util.fbpmsdnet_online_reconstructor import FBPMSDNetReconstructor
    MSD_PYTORCH_AVAILABLE = True
except ImportError:
    MSD_PYTORCH_AVAILABLE = False
try:
    from util.cinn_reconstructor import CINNReconstructor
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import loggers as pl_loggers
    CINN_AVAILABLE = True
except ImportError:
    CINN_AVAILABLE = False
# from util.fbpmsdnet_original_reconstructor import (
#     FBPMSDNetOriginalReconstructor)
from util.apples_dataset import get_apples_dataset

# torch.backends.cudnn.benchmark = True

IMPL = 'astra_cuda'
RESULTS_PATH = '../learned_ct_reco_comparison_paper_results'

NOISE_SETTING_DEFAULT = 'gaussian_noise'
NUM_ANGLES_DEFAULT = 50
METHOD_DEFAULT = 'learnedpd'

parser = argparse.ArgumentParser()
parser.add_argument('--noise_setting', type=str, default=NOISE_SETTING_DEFAULT)
parser.add_argument('--num_angles', type=int, default=NUM_ANGLES_DEFAULT)
parser.add_argument('--method', type=str, default=METHOD_DEFAULT)

options = parser.parse_args()

noise_setting = options.noise_setting  # 'gaussian_noise', 'scattering'
num_angles = options.num_angles  # 50, 10, 5, 2
method = options.method  # 'learnedpd', 'fbpunet', 'fbpmsdnet', 'cinn'
name = 'apples_{}_{:02d}_{}'.format(noise_setting, num_angles, method)

dataset = get_apples_dataset(num_angles=num_angles,
                             noise_setting=noise_setting,
                             impl=IMPL)

# # only use subset of data (e.g. for testing code):
# dataset.train_len = 1
# dataset.validation_len = 1

FBP_DATASET_STATS = {
    'noisefree': {
        2: {
            'mean_fbp': 0.0020300781237049294,
            'std_fbp': 0.0036974098858769781,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        },
        5: {
            'mean_fbp': 0.0018914765285141003,
            'std_fbp': 0.0027988724415204552,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        },
        10: {
            'mean_fbp': 0.0018791806499857538,
            'std_fbp': 0.0023355593815585413,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        },
        50: {
            'mean_fbp': 0.0018856220845133943,
            'std_fbp': 0.002038545754978578,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        }
    },
    'gaussian_noise': {
        2: {
            'mean_fbp': 0.0020300515246877825,
            'std_fbp': 0.01135122820016111,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        },
        5: {
            'mean_fbp': 0.0018914835384669934,
            'std_fbp': 0.0073404856822226593,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        },
        10: {
            'mean_fbp': 0.0018791781748714272,
            'std_fbp': 0.0053367740312729459,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        },
        50: {
            'mean_fbp': 0.0018856252771456445,
            'std_fbp': 0.0029598508235758759,
            'mean_gt': 0.0018248517968347585,
            'std_gt': 0.0020251920919838714
        }
    },
    'scattering': {
        2: {
            'mean_fbp': 0.68570249744436962,
            'std_fbp': 1.3499668155231217,
            'mean_gt': 0.002007653630624356,  # different from gaussian_noise
            'std_gt': 0.0019931366497635745   # since subset of slices is used
        },
        5: {
            'mean_fbp': 0.67324839540841908,
            'std_fbp': 0.99012416989800478,
            'mean_gt': 0.002007653630624356,  # different from gaussian_noise
            'std_gt': 0.0019931366497635745   # since subset of slices is used
        },
        10: {
            'mean_fbp': 0.66960775275347806,
            'std_fbp': 0.80318946689776671,
            'mean_gt': 0.002007653630624356,  # different from gaussian_noise
            'std_gt': 0.0019931366497635745   # since subset of slices is used
        },
        50: {
            'mean_fbp': 0.67173917657611049,
            'std_fbp': 0.6794825395874754,
            'mean_gt': 0.002007653630624356,  # different from gaussian_noise
            'std_gt': 0.0019931366497635745   # since subset of slices is used
        }
    }
}

ray_trafo = dataset.ray_trafo

if method == 'learnedpd':
    reconstructor = LearnedPDReconstructor(
        ray_trafo,
        hyper_params={
            "batch_size": 1,
            "epochs": 50,
            "niter": 10,
            "internal_ch": 64,
            "lr": 0.0001,
            "lr_min": 0.0001,
            "init_fbp": True,
            "init_frequency_scaling": 0.1
        },
        save_best_learned_params_path=os.path.join(RESULTS_PATH, name),
        log_dir=os.path.join(RESULTS_PATH, name),
        num_data_loader_workers=0,
        )
elif method == 'fbpunet':
    reconstructor = FBPUNetReconstructor(
        ray_trafo,
        hyper_params={
            "scales": 5,
            "skip_channels": 4,
            "batch_size": 4,
            "epochs": 50,
            "lr": 0.001,
            "filter_type": "Hann",
            "frequency_scaling": 1.0,
            "init_bias_zero": True,
            "scheduler": "cosine",
            "lr_min": 0.001,
            'norm_type': 'layer'
        },
        save_best_learned_params_path=os.path.join(RESULTS_PATH, name),
        log_dir=os.path.join(RESULTS_PATH, name),
        num_data_loader_workers=0,
        )
elif method == 'fbpmsdnet':
    assert MSD_PYTORCH_AVAILABLE
    # reconstructor = FBPMSDNetOriginalReconstructor(
    reconstructor = FBPMSDNetReconstructor(
        ray_trafo,
        hyper_params={
            'depth': 100,
            'width': 1,
            'dilations': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            'lr': 0.001,
            'batch_size': 1,
            'epochs': 50,
            'data_augmentation': True,
            'scheduler': 'none'
        },
        save_best_learned_params_path=os.path.join(RESULTS_PATH, name),
        log_dir=os.path.join(RESULTS_PATH, name),
        num_data_loader_workers=0,
        )
    dataset.fbp_dataset_stats = FBP_DATASET_STATS[noise_setting][num_angles]
elif method == 'cinn':
    assert CINN_AVAILABLE 
    checkpoint_callback = ModelCheckpoint(filepath=None, save_top_k=-1,
                                          verbose=True, monitor='val_loss',
                                          mode='min', prefix='')
    log_dir=os.path.join(RESULTS_PATH, name)   
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)
    reconstructor = CINNReconstructor(
        ray_trafo=ray_trafo,
        trainer_args={'distributed_backend': 'ddp',
                      'gpus': -1,
                      'default_root_dir': log_dir,
                      'benchmark': True,
                      'gradient_clip_val': 0.01,
                      'logger': tb_logger,
                      'precision': 16
                      },
        hyper_params={
            'lr': 0.0005,
            'weight_decay': 0.0,
            'batch_size': 3,
            'epochs': 200,
            'torch_manual_seed': None,
            'weight_mse': 1.0,
        })
else:
    raise ValueError("unknown reconstructor '{}'".format(method))

reconstructor.save_hyper_params(
    os.path.join(RESULTS_PATH, '{}_hyper_params.json'.format(name)))

print("start training: '{}'".format(name))
print('hyper_params = {}'.format(
    json.dumps(reconstructor.hyper_params, indent=1)))
reconstructor.train(dataset)
