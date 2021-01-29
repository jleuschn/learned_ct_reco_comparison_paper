# -*- coding: utf-8 -*-
"""
Data loader for the secret test set of the Apple CT data.
"""
import os
from warnings import warn
import numpy as np
from odl import (
    uniform_discr, nonuniform_partition, uniform_discr_frompartition)
from odl.tomo import parallel_beam_geometry, Parallel2dGeometry, RayTransform
import tifffile
from PIL import Image

config = {'data_path': '/localdata/jleuschn/AppleCT_Test'}

NUM_IMAGES = 13500
NUM_IMAGES_SCATTERING = 1600
NUM_APPLES = 20
IMAGE_SHAPE = (972, 972)
_IMAGE_SHAPE = (1000, 1000)
NUM_ANGLES = 50
NUM_DET_PIXELS = 1377
MIN_PT = [-1.0, -1.0]
MAX_PT = [1.0, 1.0]


image_domain = uniform_discr(
    min_pt=MIN_PT, max_pt=MAX_PT, shape=IMAGE_SHAPE, dtype=np.float32)
label_domain = uniform_discr(
    min_pt=MIN_PT, max_pt=MAX_PT, shape=IMAGE_SHAPE, dtype=np.uint8)
geometry = parallel_beam_geometry(
    image_domain, num_angles=NUM_ANGLES)

obs_domain = uniform_discr(geometry.partition.min_pt,
                           geometry.partition.max_pt,
                           (NUM_ANGLES, NUM_DET_PIXELS), dtype=np.float32)

ground_truth_files = []
files = os.listdir(os.path.join(config['data_path'], 'recons'))
files.sort()
for file in files:
    ground_truth_files.append(os.path.join('recons', file))
slice_ids = [f[7:16] for f in ground_truth_files]
observation_files = {}
for p in ('gaussian_noise', 'noisefree', 'scattering'):
    try:
        observation_files[p] = os.listdir(
            os.path.join(config['data_path'], 'projections_{}'.format(p)))
        observation_files[p].sort()
    except FileNotFoundError:
        warn("missing projection files '{}'".format(p))
label_files = os.listdir(os.path.join(config['data_path'], 'labels'))
label_files.sort()
apple_ids = np.unique([int(f[:5]) for f in label_files])
if 'scattering' in observation_files:
    scattering_indices = []
    _i = 0
    for _f in observation_files['scattering']:
        while _f[5:-4] != label_files[_i][:-4]:
            _i += 1
        scattering_indices.append(_i)
else:
    warn('scattering files are missing, using indices into the scattering '
         'subset (i.e. calling data loader functions with `scattering=True`) '
         'will not work')

def get_obs_domain(num_angles=50):
    if num_angles == 50:
        return obs_domain
    else:
        assert NUM_ANGLES % num_angles == 0
        apart = nonuniform_partition(
            geometry.angles[
                (NUM_ANGLES//num_angles)//2:NUM_ANGLES:NUM_ANGLES//num_angles])
        subset_obs_domain = uniform_discr_frompartition(
            apart.append(geometry.det_partition), dtype=obs_domain.dtype)
        return subset_obs_domain

def get_ray_trafo(num_angles=50, impl='astra_cuda'):
    if num_angles == 50:
        ray_trafo = RayTransform(image_domain, geometry, impl=impl)
    else:
        assert NUM_ANGLES % num_angles == 0
        apart = nonuniform_partition(
            geometry.angles[
                (NUM_ANGLES//num_angles)//2:NUM_ANGLES:NUM_ANGLES//num_angles])
        subset_geometry = Parallel2dGeometry(
            apart=apart, dpart=geometry.det_partition)
        ray_trafo = RayTransform(image_domain, subset_geometry, impl=impl)
    return ray_trafo

def get_ground_truth(idx, scattering=False, out=None):
    """
    Return a ground truth sample from the apples-ct training dataset.

    Parameters
    ----------
    idx : int
        Index of the sample.
    scattering : bool, optional
        Whether `idx` is interpreted as an index into the scattering subset.
        The default is `False`.
    out : odl element or array, optional
        Array in which to store the observation.
        Must have shape ``(972, 972)``.
        If `None`, a new odl element is created.

    Raises
    ------
    IndexError
        If `idx` is out of bounds.

    Returns
    -------
    out : odl element or array
        Array holding the ground truth.
    """
    n = NUM_IMAGES_SCATTERING if scattering else NUM_IMAGES
    if idx >= n or idx < -n:
        raise IndexError("index {} out of bounds ({:d})".format(idx, n))
    if idx < 0:
        idx += n
    if scattering:
        idx = scattering_indices[idx]
    if out is None:
        out = image_domain.zero()
    out[:] = tifffile.imread(os.path.join(config['data_path'],
                                          ground_truth_files[idx]))
    return out

def get_observation(idx, scattering=False, noise_setting='gaussian_noise',
                    num_angles=50, out=None):
    """
    Return an observation sample from the apples-ct training dataset.

    Parameters
    ----------
    idx : int
        Index of the sample.
    scattering : bool, optional
        Whether `idx` is interpreted as an index into the scattering subset.
        The default is `False`.
    noise_setting : {``'gaussian_noise'``, ``'noisefree'``, ``'scattering'``}
        Noise setting.
    num_angles : int, optional
        Number of angles. Must be an integer divisor of `50`.
        Default: `50`.
    out : odl element or array, optional
        Array in which to store the observation.
        Must have shape ``(num_angles, 1377)``.
        If `None`, a new odl element is created.

    Raises
    ------
    IndexError
        If `idx` is out of bounds.

    Returns
    -------
    out : odl element or array
        Array holding the observation.
    """
    n = NUM_IMAGES_SCATTERING if scattering else NUM_IMAGES
    if idx >= n or idx < -n:
        raise IndexError("index {} out of bounds ({:d})"
                         .format(idx, n))
    assert NUM_ANGLES % num_angles == 0
    if idx < 0:
        idx += n
    if scattering and noise_setting != 'scattering':
        idx = scattering_indices[idx]
    elif not scattering and noise_setting == 'scattering':
        idx = scattering_indices.index(idx)
    if out is None:
        out = get_obs_domain(num_angles).zero()
    obs = tifffile.imread(os.path.join(
        config['data_path'], 'projections_{}'.format(noise_setting),
        observation_files[noise_setting][idx]))
    out[:] = (
        obs if num_angles == NUM_ANGLES else
        obs[(NUM_ANGLES//num_angles)//2:NUM_ANGLES:NUM_ANGLES//num_angles])
    return out

def get_labels(idx, scattering=False, map_labels=None, out=None):
    """
    Return a segmentation label sample from the apples-ct test dataset.

    Parameters
    ----------
    idx : int
        Index of the sample.
    scattering : bool, optional
        Whether `idx` is interpreted as an index into the scattering subset.
        The default is `False`.
    map_labels : array-like, optional
        New labels to which the original labels [0, 64, 128, 191, 255] will be
        mapped, e.g. ``[0, 1, 1, 1, 1]`` to map all defects to one class.
        Default is ``[0, 1, 2, 3, 4]``.
    out : odl element or array, optional
        Array in which to store the labels.
        Must have shape ``(972, 972)``.
        If `None`, a new odl element is created.

    Raises
    ------
    IndexError
        If `idx` is out of bounds.

    Returns
    -------
    out : odl element or array
        Array holding the labels.
    """
    n = NUM_IMAGES_SCATTERING if scattering else NUM_IMAGES
    if idx >= n or idx < -n:
        raise IndexError("index {} out of bounds ({:d})"
                         .format(idx, n))
    if idx < 0:
        idx += n
    if map_labels is None:
        map_labels = np.arange(5)
    if scattering:
        idx = scattering_indices[idx]
    if out is None:
        out = label_domain.zero()
    labels = np.array(Image.open(os.path.join(
        config['data_path'], 'labels', label_files[idx])))
    out[labels==0] = map_labels[0]
    out[labels==64] = map_labels[1]
    out[labels==128] = map_labels[2]
    out[labels==191] = map_labels[3]
    out[labels==255] = map_labels[4]
    return out

def get_slice_indices(apple_indices, scattering=False, skip_border=0):
    """
    Return indices into the list of all available slices (which depend on the
    value of `scattering`) belonging to given apples.

    For ``scattering=False``, the indices fit for the file lists

        * `ground_truth_files`
        * `label_files`
        * `observation_files['gaussian_noise']`
        * `observation_files['noisefree']`

    while for ``scattering=True``, the indices fit for the file list

        * `observation_files['scattering']`

    Parameters
    ----------
    apple_indices : iterable of int
        The apple indices (in range(20)) to which the slices belong.
    scattering : bool, optional
        Whether the indices should match the file list
        `observation_files['scattering']` rather than the full file lists.
        Default: ``False``.
    skip_border : int, optional
        Amount of slices to skip at both ends of the scans.
        This option is not supported when using ``scattering=True``.
        Default: ``0``.
    """
    ids = [apple_ids[i] for i in apple_indices]
    assert not (scattering and skip_border > 0)
    if scattering:
        slice_indices_per_apple = [
            [i for i, f in enumerate(observation_files['scattering'])
             if int(f[5:10]) == id_]
            for id_ in ids]
    else:
        slice_indices_per_apple = [
            [i for i, f in enumerate(observation_files['gaussian_noise'])
             if int(f[11:16]) == id_]
            for id_ in ids]
    slice_indices = []
    for s in slice_indices_per_apple:
        slice_indices += s[skip_border:len(s)-skip_border]
    return slice_indices

def get_slice_id(idx, scattering=False):
    """
    Return slice id (format `XXXXX_XXX`), specifying apple and slice.

    Parameters
    ----------
    idx : int
        Index of the sample.
    scattering : bool, optional
        Whether `idx` is interpreted as an index into the scattering subset.
        The default is `False`.

    Raises
    ------
    IndexError
        If `idx` is out of bounds.

    Returns
    -------
    slice_id : int
        Slice id.
    """
    n = NUM_IMAGES_SCATTERING if scattering else NUM_IMAGES
    if idx >= n or idx < -n:
        raise IndexError("index {} out of bounds ({:d})"
                         .format(idx, n))
    if idx < 0:
        idx += n
    if scattering:
        idx = scattering_indices[idx]
    return slice_ids[idx]

def get_index_by_slice_id(slice_id, scattering=False):
    """
    Return index from slice id.

    Parameters
    ----------
    slice_id : str
        Index of the sample.
    scattering : bool, optional
        Whether the returned index should be an index into the scattering
        subset.
        The default is `False`.

    Raises
    ------
    ValueError
        If `slice_id` is invalid.

    Returns
    -------
    idx : int
        Index into either the full or the scattering subset, depending on the
        value of `scattering`.
    """
    idx = slice_ids.index(slice_id)
    if scattering:
        idx = scattering_indices.index(idx)
    return idx
