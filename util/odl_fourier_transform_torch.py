# -*- coding: utf-8 -*-
"""
Provides a torch :class:`Module` computing the fourier transform like ODL does.

Note that this transform differs from a standard DFT because shifted
frequencies are used (i.e. the zero frequency is not included) and a
post-processing weighting is applied. Cf.
`https://odlgroup.github.io/odl/generated/odl.trafos.fourier.FourierTransform.html`_.
"""
import numpy as np
import torch
from odl.trafos.util import reciprocal_space

def _get_signal_factor(signal_domain):
    factor = torch.ones((signal_domain.shape[-1],), requires_grad=False)
    factor[1::2] = -1.
    return factor

def _get_fourier_factor(signal_domain, fourier_domain, inverse=False):
    real_grid = signal_domain.grid
    recip_grid = fourier_domain.grid
    imag = 1j if inverse else -1j
    factor = np.exp(imag * real_grid.min_pt[-1] * recip_grid.coord_vectors[-1])
    len_dft = recip_grid.shape[-1]
    len_orig = real_grid.shape[-1]
    odd = len_orig % 2
    fmin = -0.5
    fmax = -1.0 / (2 * len_orig) if odd else 0.0
    freqs = np.linspace(fmin, fmax, num=len_dft)
    stride = real_grid.stride[-1]
    interp_kernel = np.sinc(freqs) / np.sqrt(2 * np.pi) * stride
    if inverse:
        factor /= interp_kernel
    else:
        factor *= interp_kernel
    return torch.stack([torch.from_numpy(factor.real.astype(np.float32)),
                        torch.from_numpy(factor.imag.astype(np.float32))],
                       dim=-1)

class FourierTransformModule(torch.nn.Module):
    """
    Clone of `odl.trafos.fourier.FourierTransform` using torch
    for a 1-dimensional transform using the last axis and default arguments.
    """
    def __init__(self, signal_domain):
        super().__init__()
        self.signal_domain = signal_domain
        self.fourier_domain = reciprocal_space(
            self.signal_domain, axes=(self.signal_domain.ndim-1,),
            halfcomplex=True)
        self.preproc_factor = _get_signal_factor(self.signal_domain)
        self.postproc_factor = _get_fourier_factor(self.signal_domain,
                                                   self.fourier_domain)

    def preprocess(self, x):
        return self.preproc_factor.to(x.device) * x

    def postprocess(self, y):
        f = self.postproc_factor.to(y.device)
        return torch.stack([f[..., 0] * y[..., 0] - f[..., 1] * y[..., 1],
                            f[..., 0] * y[..., 1] + f[..., 1] * y[..., 0]],
                           dim=-1)

    def forward(self, x):
        x_preproc = self.preprocess(x)
        y_nonpostproc = torch.rfft(x_preproc, 1)
        y = self.postprocess(y_nonpostproc)
        return y

class FourierTransformInverseModule(torch.nn.Module):
    """
    Clone of `odl.trafos.fourier.FourierTransformInverse` using torch
    for a 1-dimensional transform using the last axis and default arguments.
    """
    def __init__(self, signal_domain):
        super().__init__()
        self.signal_domain = signal_domain
        self.fourier_domain = reciprocal_space(
            self.signal_domain, axes=(self.signal_domain.ndim-1,),
            halfcomplex=True)
        self.preproc_factor = _get_fourier_factor(self.signal_domain,
                                                  self.fourier_domain,
                                                  inverse=True)
        self.postproc_factor = _get_signal_factor(self.signal_domain)

    def preprocess(self, y):
        f = self.preproc_factor.to(y.device)
        return torch.stack([f[..., 0] * y[..., 0] - f[..., 1] * y[..., 1],
                            f[..., 0] * y[..., 1] + f[..., 1] * y[..., 0]],
                           dim=-1)

    def postprocess(self, x):
        return self.postproc_factor.to(x.device) * x

    def forward(self, y):
        y_preproc = self.preprocess(y)
        x_nonpostproc = torch.irfft(
            y_preproc, 1, signal_sizes=(self.signal_domain.shape[-1],))
        x = self.postprocess(x_nonpostproc)
        return x
