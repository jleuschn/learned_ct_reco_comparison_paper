# -*- coding: utf-8 -*-
"""
MS-D network reconstructor computing FBPs on the fly.

The implementation includes rotate-and-flip data augmentation.

Means and standard deviations of input and output data need to be passed to the
`train` method via attribute ``dataset.fbp_dataset_stats``.
"""
from copy import deepcopy
from math import ceil

import torch
import numpy as np
import torch.nn as nn
from odl.tomo import fbp_op
from tqdm import tqdm

from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    TENSORBOARD_AVAILABLE = False
else:
    TENSORBOARD_AVAILABLE = True
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR

from dival.reconstructors.standard_learned_reconstructor import (
    StandardLearnedReconstructor)
from dival.measure import PSNR
from util.msdnet import MSDNet
from util.transforms import random_flip_rotate_transform_fbp



from odl.tomo.analytic import fbp_filter_op
from odl.contrib.torch import OperatorModule
from dival.util.torch_utility import TorchRayTrafoParallel2DAdjointModule
from util.fbp_filter_module import FBPFilterModule

class FBPModule(torch.nn.Module):
    def __init__(
            self, ray_trafo, filter_type='Hann', frequency_scaling=1.):
        super().__init__()
        self.ray_trafo = ray_trafo
        self.filter_mod = FBPFilterModule(self.ray_trafo,
                                          filter_type=filter_type,
                                          frequency_scaling=frequency_scaling)
        # filter_op = fbp_filter_op(self.ray_trafo,
        #                           filter_type=filter_type,
        #                           frequency_scaling=frequency_scaling)
        # self.filter_mod = OperatorModule(filter_op)
        self.ray_trafo_adjoint_mod = (
            TorchRayTrafoParallel2DAdjointModule(self.ray_trafo))
        # self.ray_trafo_adjoint_mod = (
        #     OperatorModule(self.ray_trafo.adjoint))

    def forward(self, x):
        x = self.filter_mod(x)
        x = self.ray_trafo_adjoint_mod(x)
        return x



def compute_fbp_dataset_stats(fbp_dataset):
    """
    Compute means and standard deviations for the elements of an FBP dataset.
    Only the ``'train'`` part is used.
    """
    # Adapted from: https://github.com/ahendriksen/msd_pytorch/blob/162823c502701f5eedf1abcd56e137f8447a72ef/msd_pytorch/msd_model.py#L95
    mean_fbp = 0.
    mean_gt = 0.
    square_fbp = 0.
    square_gt = 0.
    n = fbp_dataset.get_len('train')
    for fbp, gt in tqdm(fbp_dataset.generator('train'), total=n,
                        desc='computing fbp dataset stats'):
        mean_fbp += np.mean(fbp)
        mean_gt += np.mean(gt)
        square_fbp += np.mean(np.square(fbp))
        square_gt += np.mean(np.square(gt))
    mean_fbp /= n
    mean_gt /= n
    square_fbp /= n
    square_gt /= n
    std_fbp = np.sqrt(square_fbp - mean_fbp**2)
    std_gt = np.sqrt(square_gt - mean_gt**2)
    stats = {'mean_fbp': mean_fbp,
             'std_fbp': std_fbp,
             'mean_gt': mean_gt,
             'std_gt': std_gt}
    return stats

class FBPMSDNetReconstructor(StandardLearnedReconstructor):
    """
    CT reconstructor applying filtered back-projection followed by a
    postprocessing U-Net (e.g. [1]_).

    References
    ----------
    .. [1] K. H. Jin, M. T. McCann, E. Froustey, et al., 2017,
           "Deep Convolutional Neural Network for Inverse Problems in Imaging".
           IEEE Transactions on Image Processing.
           `doi:10.1109/TIP.2017.2713099
           <https://doi.org/10.1109/TIP.2017.2713099>`_
    """

    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'depth': {
            'default': 100,
            'retrain': True
        },
        'width': {
            'default': 1,
            'retrain': True
        },
        'dilations': {
            'default': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            'retrain': True
        },
        'filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'frequency_scaling': {
            'default': 1.0,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'scheduler': {
            'default': 'none',
            'choices': ['none', 'base', 'cosine'],  # 'base': inherit
            'retrain': True
        },
        'lr_min': {  # only used if 'cosine' scheduler is selected
            'default': 1e-4,
            'retrain': True
        },
        'data_augmentation': {
            'default': True,
            'retrain': True
        }
    })

    def __init__(self, ray_trafo, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform (the forward operator).

        Further keyword arguments are passed to ``super().__init__()``.
        """
        self._fbp_dataset_stats = None
        super().__init__(ray_trafo, **kwargs)

    def train(self, dataset):
        if self.torch_manual_seed:
            torch.random.manual_seed(self.torch_manual_seed)

        # create PyTorch datasets
        dataset_train = dataset.create_torch_dataset(
            part='train', reshape=((1,) + dataset.space[0].shape,
                                   (1,) + dataset.space[1].shape))

        dataset_validation = dataset.create_torch_dataset(
            part='validation', reshape=((1,) + dataset.space[0].shape,
                                        (1,) + dataset.space[1].shape))

        try:
            self._fbp_dataset_stats = dataset.fbp_dataset_stats
        except AttributeError:
            raise ValueError('Please set the attribute '
                  "``dataset.fbp_dataset_stats = {"
                  "    'mean_fbp': ..., "
                  "    'std_fbp': ..., "
                  "    'mean_gt': ..., "
                  "    'std_gt': ...}``."
                  'This dict can be computed using '
                  '``compute_fbp_dataset_stats(fbp_dataset)``.')

        # reset model before training
        self.init_model()

        self._fbp_dataset_stats = None  # reset, because the only purpose is to
                                        # expose the stats to self.init_model()

        criterion = torch.nn.MSELoss()
        self.init_optimizer(dataset_train=dataset_train)

        # create PyTorch dataloaders
        data_loaders = {'train': DataLoader(
            dataset_train, batch_size=self.batch_size,
            num_workers=self.num_data_loader_workers, shuffle=True,
            pin_memory=True),
            'validation': DataLoader(
                dataset_validation, batch_size=self.batch_size,
                num_workers=self.num_data_loader_workers,
                shuffle=True, pin_memory=True)}

        dataset_sizes = {'train': len(dataset_train),
                         'validation': len(dataset_validation)}

        self.init_scheduler(dataset_train=dataset_train)
        if self._scheduler is not None:
            schedule_every_batch = isinstance(
                self._scheduler, (CyclicLR, OneCycleLR))

        best_model_wts = deepcopy(self.model.state_dict())
        best_psnr = 0

        if self.log_dir is not None:
            if not TENSORBOARD_AVAILABLE:
                raise ImportError(
                    'Missing tensorboard. Please install it or disable '
                    'logging by specifying `log_dir=None`.')
            writer = SummaryWriter(log_dir=self.log_dir, max_queue=0)
            validation_samples = dataset.get_data_pairs(
                'validation', self.log_num_validation_samples)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            # Each epoch has a training and validation phase
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_psnr = 0.0
                running_loss = 0.0
                running_size = 0
                with tqdm(data_loaders[phase],
                          desc='epoch {:d}'.format(epoch + 1),
                          disable=not self.show_pbar) as pbar:
                    for inputs, labels in pbar:
                        if self.normalize_by_opnorm:
                            inputs = (1./self.opnorm) * inputs
                        inputs = inputs.to(self.device)
                        with torch.no_grad():
                            inputs = self.fbp_module(inputs)
                        labels = labels.to(self.device)
                        if self.data_augmentation:
                            inputs, labels = random_flip_rotate_transform_fbp(
                                (inputs, labels), dims=(2, 3))

                        # zero the parameter gradients
                        self._optimizer.zero_grad()

                        # forward
                        # track gradients only if in train phase
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), max_norm=1)
                                self._optimizer.step()
                                if (self._scheduler is not None and
                                        schedule_every_batch):
                                    self._scheduler.step()

                        for i in range(outputs.shape[0]):
                            labels_ = labels[i, 0].detach().cpu().numpy()
                            outputs_ = outputs[i, 0].detach().cpu().numpy()
                            running_psnr += PSNR(outputs_, labels_)

                        # statistics
                        running_loss += loss.item() * outputs.shape[0]
                        running_size += outputs.shape[0]

                        pbar.set_postfix({'phase': phase,
                                          'loss': running_loss/running_size,
                                          'psnr': running_psnr/running_size})
                        if self.log_dir is not None and phase == 'train':
                            step = (epoch * ceil(dataset_sizes['train']
                                                 / self.batch_size)
                                    + ceil(running_size / self.batch_size))
                            writer.add_scalar(
                                'loss/{}'.format(phase),
                                torch.tensor(running_loss/running_size), step)
                            writer.add_scalar(
                                'psnr/{}'.format(phase),
                                torch.tensor(running_psnr/running_size), step)

                    if (self._scheduler is not None
                            and not schedule_every_batch):
                        self._scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_psnr = running_psnr / dataset_sizes[phase]

                    if self.log_dir is not None and phase == 'validation':
                        step = (epoch+1) * ceil(dataset_sizes['train']
                                                / self.batch_size)
                        writer.add_scalar('loss/{}'.format(phase),
                                          epoch_loss, step)
                        writer.add_scalar('psnr/{}'.format(phase),
                                          epoch_psnr, step)

                    # deep copy the model (if it is the best one seen so far)
                    if phase == 'validation' and epoch_psnr > best_psnr:
                        best_psnr = epoch_psnr
                        best_model_wts = deepcopy(self.model.state_dict())
                        if self.save_best_learned_params_path is not None:
                            self.save_learned_params(
                                self.save_best_learned_params_path)

                if (phase == 'validation' and self.log_dir is not None and
                        self.log_num_validation_samples > 0):
                    with torch.no_grad():
                        val_images = []
                        for (y, x) in validation_samples:
                            y = torch.from_numpy(
                                np.asarray(y))[None, None].to(self.device)
                            x = torch.from_numpy(
                                np.asarray(x))[None, None].to(self.device)
                            reco = self.model(y)
                            reco -= torch.min(reco)
                            reco /= torch.max(reco)
                            val_images += [reco, x]
                        writer.add_images(
                            'validation_samples', torch.cat(val_images),
                            (epoch + 1) * (ceil(dataset_sizes['train'] /
                                                self.batch_size)),
                            dataformats='NCWH')

        print('Best val psnr: {:4f}'.format(best_psnr))
        self.model.load_state_dict(best_model_wts)

    def init_model(self):
        self.fbp_op = fbp_op(self.op, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)
        self.fbp_module = FBPModule(self.op,
                                    filter_type=self.filter_type,
                                    frequency_scaling=self.frequency_scaling)
        self.model = MSDNet(in_ch=1, out_ch=1, depth=self.depth,
                            width=self.width, dilations=self.dilations)
        if self._fbp_dataset_stats is not None:
            self.model.set_normalization(
                mean_in=self._fbp_dataset_stats['mean_fbp'],
                std_in=self._fbp_dataset_stats['std_fbp'],
                mean_out=self._fbp_dataset_stats['mean_gt'],
                std_out=self._fbp_dataset_stats['std_gt'])

        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    def init_optimizer(self, dataset_train):
        """
        Initialize the optimizer.
        Called in :meth:`train`, after calling :meth:`init_model` and before
        calling :meth:`init_scheduler`.

        Parameters
        ----------
        dataset_train : :class:`torch.utils.data.Dataset`
            The training (torch) dataset constructed in :meth:`train`.
        """
        # only train msd, but not scale_in and scale_out
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)

    def init_scheduler(self, dataset_train):
        # need to set private self._scheduler because self.scheduler
        # property accesses hyper parameter of same name,
        # i.e. self.hyper_params['scheduler']
        if self.scheduler.lower() == 'none':
            self._scheduler = None
        elif self.scheduler.lower() == 'cosine':
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.lr_min)
        else:
            super().init_scheduler(dataset_train)

    def _reconstruct(self, observation):
        self.model.eval()
        fbp = self.fbp_op(observation)
        fbp_tensor = torch.from_numpy(
            np.asarray(fbp)[None, None]).to(self.device)
        reco_tensor = self.model(fbp_tensor)
        reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)
