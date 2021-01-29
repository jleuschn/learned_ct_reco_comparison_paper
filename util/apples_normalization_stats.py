# -*- coding: utf-8 -*-
"""
Compute means and standard deviations for FBPs and ground truths of the Apple
CT training data.
"""
import numpy as np
from tqdm import tqdm
from util.apples_dataset import get_apples_dataset
from dival.datasets.fbp_dataset import FBPDataset

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

def compute_fbp_dataset_stats_only_fbp(fbp_dataset):
    """
    Compute means and standard deviations for the elements of an FBP dataset.
    Only the ``'train'`` part is used.
    """
    # Adapted from: https://github.com/ahendriksen/msd_pytorch/blob/162823c502701f5eedf1abcd56e137f8447a72ef/msd_pytorch/msd_model.py#L95
    mean_fbp = 0.
    square_fbp = 0.
    n = fbp_dataset.get_len('train')
    for i in tqdm(range(n),
                        desc='computing fbp dataset stats'):
        fbp, _ = fbp_dataset.get_sample(i, out=(True, False))
        mean_fbp += np.mean(fbp)
        square_fbp += np.mean(np.square(fbp))
    mean_fbp /= n
    square_fbp /= n
    std_fbp = np.sqrt(square_fbp - mean_fbp**2)
    stats = {'mean_fbp': mean_fbp,
             'std_fbp': std_fbp}
    return stats

if __name__ == '__main__':
    for noise_setting in ['scattering', 'gaussian_noise']:
        for num_angles in [2, 5, 10, 50]:
            dataset = get_apples_dataset(num_angles=num_angles,
                                         noise_setting=noise_setting)
            fbp_dataset = FBPDataset(dataset, dataset.ray_trafo)
            if num_angles == 50:
                stats = compute_fbp_dataset_stats(fbp_dataset)
            else:
                stats = compute_fbp_dataset_stats_only_fbp(fbp_dataset)
            print(stats)
