# -*- coding: utf-8 -*-
"""
Provides a torch :class:`Module` computing the filtering step of the filtered
back-projection like ODL does.
"""
import numpy as np
from odl import ResizingOperator
import torch
import torch.nn.functional as F
from util.odl_fourier_transform_torch import (
    FourierTransformModule, FourierTransformInverseModule)

# function taken from odl.tomo.analytic.filtered_back_projection
def _fbp_filter(norm_freq, filter_type, frequency_scaling):
    filter_type, filter_type_in = str(filter_type).lower(), filter_type
    if callable(filter_type):
        filt = filter_type(norm_freq)
    elif filter_type == 'ram-lak':
        filt = np.copy(norm_freq)
    elif filter_type == 'shepp-logan':
        filt = norm_freq * np.sinc(norm_freq / (2 * frequency_scaling))
    elif filter_type == 'cosine':
        filt = norm_freq * np.cos(norm_freq * np.pi / (2 * frequency_scaling))
    elif filter_type == 'hamming':
        filt = norm_freq * (
            0.54 + 0.46 * np.cos(norm_freq * np.pi / (frequency_scaling)))
    elif filter_type == 'hann':
        filt = norm_freq * (
            np.cos(norm_freq * np.pi / (2 * frequency_scaling)) ** 2)
    else:
        raise ValueError('unknown `filter_type` ({})'
                         ''.format(filter_type_in))

    indicator = (norm_freq <= frequency_scaling)
    filt *= indicator
    return filt

class FBPFilterModule(torch.nn.Module):
    def __init__(self, ray_trafo, filter_type='Hann', frequency_scaling=1.0):
        super().__init__()
        self.ray_trafo = ray_trafo
        obs_space = self.ray_trafo.range
        self.device = (torch.device('cuda:0')
                       if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.pad_op = ResizingOperator(obs_space,
                                       ran_shp=(obs_space.shape[0],
                                                obs_space.shape[1]*2-1))
        self.fourier_mod = FourierTransformModule(self.pad_op.range)
        self.fourier_inverse_mod = FourierTransformInverseModule(
            self.pad_op.range)
    
        def fourier_filter(x):
            abs_freq = np.abs(x[1])
            norm_freq = abs_freq / np.max(abs_freq)
            filt = _fbp_filter(norm_freq, filter_type, frequency_scaling)
            scaling = 1. / (2. * np.pi)
            return filt * np.max(abs_freq) * scaling
        
        self.ramp_function = (
            torch.from_numpy(
                fourier_filter((None,
                                self.fourier_mod.fourier_domain.meshgrid[1])))
            .to(dtype=torch.float32, device=self.device)[..., None])

    def forward(self, x):
        obs_space = self.ray_trafo.range
        pad_offset = (obs_space.shape[1]-1)//2
        x = F.pad(x, (pad_offset, obs_space.shape[1]-1-pad_offset))
        x_f = self.fourier_mod(x)
        x_filtered_f = self.ramp_function * x_f
        x_filtered = self.fourier_inverse_mod(x_filtered_f)
        x_filtered = x_filtered[
            :, :, :, pad_offset:pad_offset+obs_space.shape[1]].contiguous()
        return x_filtered
