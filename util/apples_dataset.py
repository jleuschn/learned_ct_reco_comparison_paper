"""
Provides the Apple CT training and validation data in form of
a :class:`dival.Dataset`.
"""
from warnings import warn
import numpy as np
import odl
from dival.datasets.dataset import Dataset
from dival.datasets.angle_subset_dataset import AngleSubsetDataset
from util.apples_data import (
    image_domain, obs_domain, geometry, get_observation, get_ground_truth,
    get_slice_indices, get_slice_id, NUM_APPLES, NUM_ANGLES)

def get_apples_dataset(num_angles=NUM_ANGLES, **kwargs):
    """
    Return the apples dataset, possibly with fewer angles.

    The `ApplesDataset` instance can be accessed through the attribute
    `dataset.dataset` of the returned `dataset` object (the attribute is added
    to the `ApplesDataset` instance returned for `num_angles=50` for
    convenience).

    Parameters
    ----------
    num_angles : int, optional
        Number of angles. Must be an integer divisor of `50`.
        Default: `50`.

    Returns
    -------
    dataset : `ApplesDataset` or `AngleSubsetDataset`
        The apples dataset.
        If the number of angles is less than `50`, an `AngleSubsetDataset`
        wrapping the original `ApplesDataset` is returned.
    """
    apples_dataset = ApplesDataset(**kwargs)
    assert NUM_ANGLES % num_angles == 0
    if num_angles == NUM_ANGLES:
        dataset = apples_dataset
        dataset.dataset = dataset
    else:
        angle_indices = np.arange(
            (NUM_ANGLES/num_angles)//2, NUM_ANGLES, NUM_ANGLES/num_angles,
            dtype=np.int)
        dataset = AngleSubsetDataset(apples_dataset, angle_indices)
    dataset.num_angles = num_angles
    return dataset

class ApplesDataset(Dataset):
    def __init__(self, noise_setting='gaussian_noise', shuffle=True,
                 impl='astra_cuda', apple_split=None, skip_border=0, **kwargs):
        self.noise_setting = noise_setting
        self.shape = (obs_domain.shape, image_domain.shape)
        self.scattering = self.noise_setting == 'scattering'
        if apple_split is None:
            # note: the split was computed using the
            # apples_split_train_val_data_analysis.py script in order to have
            # similar defect statistics for both 'train' and 'validation' parts
            apple_split = {'validation': [11, 12, 16, 33, 36, 42, 55, 61],
                           'test': []}
        if 'train' not in apple_split:
            apple_split['train'] = [i for i in range(NUM_APPLES)
                                    if i not in apple_split['validation']
                                    and i not in apple_split['test']]
        if self.scattering and skip_border > 0:
            warn('ignoring skip_border > 0 because the scattering '
                 'projections are used')
            skip_border = 0
        self.indices = {
            'train': get_slice_indices(
                apple_split['train'], scattering=self.scattering,
                skip_border=skip_border),
            'validation': get_slice_indices(
                apple_split['validation'], scattering=self.scattering,
                skip_border=skip_border),
            'test': get_slice_indices(
                apple_split['test'], scattering=self.scattering,
                skip_border=skip_border)
        }
        if shuffle:
            rng = np.random.default_rng(1)
            rng.shuffle(self.indices['train'])
            rng.shuffle(self.indices['validation'])
            rng.shuffle(self.indices['test'])
        self.train_len = len(self.indices['train'])
        self.validation_len = len(self.indices['validation'])
        self.test_len = len(self.indices['test'])
        self.random_access = True
        self.num_elements_per_sample = 2
        self.geometry = geometry
        space = (obs_domain, image_domain)
        super().__init__(space=space)
        self.ray_trafo = self.get_ray_trafo(impl=impl)

    def get_ray_trafo(self, **kwargs):
        return odl.tomo.RayTransform(self.space[1], self.geometry, **kwargs)

    def generator(self, part='train'):
        for idx in self.indices[part]:
            obs = get_observation(idx, scattering=self.scattering,
                                  noise_setting=self.noise_setting)
            gt = get_ground_truth(idx, scattering=self.scattering)
            yield (obs, gt)

    def get_sample(self, index, part='train', out=None):
        len_part = self.get_len(part)
        if index >= len_part or index < -len_part:
            raise IndexError("index {} out of bounds for part '{}' ({:d})"
                             .format(index, part, len_part))
        if index < 0:
            index += len_part
        idx = self.indices[part][index]
        if out is None:
            out = (True, True)
        (out_observation, out_ground_truth) = out
        if out_observation is False:
            obs = None
        elif out_observation is True:
            obs = get_observation(idx, scattering=self.scattering,
                                  noise_setting=self.noise_setting)
        else:
            obs = get_observation(idx, scattering=self.scattering,
                                  noise_setting=self.noise_setting,
                                  out=out_observation)
        if out_ground_truth is False:
            gt = None
        elif out_ground_truth is True:
            gt = get_ground_truth(idx, scattering=self.scattering)
        else:
            gt = get_ground_truth(idx, scattering=self.scattering,
                                  out=out_ground_truth)
        return (obs, gt)

    def get_slice_id(self, index, part='train'):
        len_part = self.get_len(part)
        if index >= len_part or index < -len_part:
            raise IndexError("index {} out of bounds for part '{}' ({:d})"
                             .format(index, part, len_part))
        if index < 0:
            index += len_part
        idx = self.indices[part][index]
        slice_id = get_slice_id(idx, scattering=self.scattering)
        return slice_id
