# -*- coding: utf-8 -*-
"""
Compute the maximum value across all ground truth images in the Apple CT
training data.
"""
import numpy as np
from tqdm import tqdm
from util import apples_data, apples_data_test

def compute_max_ground_truth(include_secret_test_set=True):
    max_gt = 0.
    for i in tqdm(
            range(apples_data.NUM_IMAGES),
            desc='computing max ground truth, train + validation set'):
        gt = apples_data.get_ground_truth(i)
        max_gt = max(max_gt, np.max(gt))
    if include_secret_test_set:
        for i in tqdm(
                range(apples_data_test.NUM_IMAGES),
                desc='computing max ground truth, secret test set'):
            gt = apples_data_test.get_ground_truth(i)
            max_gt = max(max_gt, np.max(gt))
    return max_gt

if __name__ == '__main__':
    max_gt = compute_max_ground_truth(include_secret_test_set=True)
    print(max_gt)

# max_gt = 0.0129353  # (secret test set + train + validation)

# train + validation: 0.0129353
# train: 0.011798
