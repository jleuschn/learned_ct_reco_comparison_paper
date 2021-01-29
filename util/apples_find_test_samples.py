# -*- coding: utf-8 -*-
"""
Script for determining a subset of slices of the secret test set of the Apple
CT data that has similar defect ratios compared to the full test set.
"""
import numpy as np
from tqdm import tqdm
from util.apples_data_test import (
    NUM_APPLES, NUM_IMAGES, NUM_IMAGES_SCATTERING, IMAGE_SHAPE,
    scattering_indices, observation_files, get_labels, get_slice_indices)

def slices_proposal(num_slices=5, min_distance=10):
    NUM_TOTAL_SLICES_PER_APPLE = 80
    # idea adapted from https://stackoverflow.com/a/49948613/4673641
    # distribute "slack"
    # (amounts by which the distances are larger than minimum distance)
    slack_inds = np.random.randint(
        num_slices+1,  # including spaces before first and after last
        size=(NUM_TOTAL_SLICES_PER_APPLE - min_distance*(num_slices-1) - 1,))
    slacks = [np.count_nonzero(slack_inds == i)
              for i in range(num_slices)]
    slices = np.arange(0, num_slices*min_distance, min_distance,
                       dtype=np.int) + np.cumsum(slacks)
    return slices

def test_samples_proposal(num_slices_per_apple=5, min_distance=10):
    test_samples = []
    for i in range(NUM_APPLES):
        slice_indices_apple = get_slice_indices([i], scattering=True)
        slices = slices_proposal(num_slices=num_slices_per_apple,
                                 min_distance=min_distance)
        test_samples += [slice_indices_apple[s] for s in slices]
    return test_samples

if __name__ == '__main__':

    # total_num_pixels = np.zeros(5, dtype=np.int64)
    # for i in range(NUM_IMAGES):
    #     unique, counts = np.unique(get_labels(i), return_counts=True)
    #     for l, c in zip(unique, counts):
    #         total_num_pixels[l] += c
    total_num_pixels = np.array([
        12698183004, 5466908, 97866, 173764, 50662458])

    # only consider test samples with scattering observations
    labels = np.zeros((NUM_IMAGES_SCATTERING,) + IMAGE_SHAPE, dtype=np.uint8)
    for i in range(NUM_IMAGES_SCATTERING):
        get_labels(i, scattering=True, out=labels[i])
    NUM_TEST_SAMPLES = 100
    NUM_TRIES = 10000
    NUM_SLICES_PER_APPLE = 5
    MIN_SLICE_DISTANCE = 10
    indices = np.arange(NUM_IMAGES_SCATTERING, dtype=np.int)
    best_test_samples = None
    best_loss = np.inf
    # min_distance_rejection_count = 0
    for _ in tqdm(range(NUM_TRIES)):

        # np.random.shuffle(indices)
        # test_samples = indices[:NUM_TEST_SAMPLES]
        test_samples = test_samples_proposal(
            num_slices_per_apple=NUM_SLICES_PER_APPLE,
            min_distance=MIN_SLICE_DISTANCE)

        test_num_pixels = np.zeros(5, dtype=np.int64)
        for l in range(1, 5):
            test_num_pixels[l] += np.count_nonzero(labels[test_samples] == l)
        ratio = ((test_num_pixels[1:] / NUM_TEST_SAMPLES)
                  / (total_num_pixels[1:] / NUM_IMAGES))
        loss = np.sum((1.-1./ratio)**2)
        if loss < best_loss:
            # slice_identifiers = [observation_files['scattering'][i][5:14]
            #       for i in test_samples]  # identifiers look like '31108_195'
            # slice_identifiers.sort()
            # reject = False
            # for i in range(1, NUM_TEST_SAMPLES):
            #     if (slice_identifiers[i][:5] == slice_identifiers[i-1][:5]
            #         and (
            #             int(slice_identifiers[i][-3:]) -
            #             int(slice_identifiers[i-1][-3:])
            #             < MIN_SLICE_DISTANCE)):
            #         reject = True  # slices in same apple are too close
            #         break
            # if reject:
            #     min_distance_rejection_count += 1
            #     continue
            best_test_samples = test_samples
            best_loss = loss
            print('new best test samples:', list(test_samples))
            print('ratio:', ratio)
            print('loss:', loss)

    # print('{:d} (of a total of {:d}) tries were rejected due to violation of '
    #       'minimum slice distance {:d}'.format(
    #           min_distance_rejection_count, NUM_TRIES, MIN_SLICE_DISTANCE))

    test_indices_scattering = best_test_samples
    # convert to non-scattering indices
    test_indices = np.array(scattering_indices)[test_indices_scattering]
    print('test_indices_scattering', list(test_indices_scattering))
    print('test_indices', list(test_indices))


# result from run with MIN_SLICE_DISTANCE=15, NUM_TRIES=10000:
# ratio: [ 1.17400366  1.08009932  1.08457448  1.19147575]
# loss: 0.0593737371447
# test_indices_scattering = [3, 21, 37, 58, 75, 86, 102, 120, 139, 155, 160, 178, 201, 221, 238, 242, 258, 276, 293, 314, 322, 341, 358, 376, 395, 401, 419, 439, 455, 473, 485, 501, 522, 538, 555, 562, 581, 603, 620, 637, 641, 661, 680, 698, 715, 723, 741, 759, 776, 794, 805, 823, 840, 857, 874, 881, 898, 915, 932, 956, 964, 986, 1003, 1021, 1038, 1042, 1059, 1080, 1095, 1115, 1121, 1139, 1161, 1179, 1195, 1204, 1224, 1241, 1260, 1275, 1284, 1300, 1321, 1339, 1357, 1362, 1379, 1398, 1416, 1435, 1442, 1459, 1478, 1500, 1518, 1521, 1542, 1562, 1580, 1597]
# test_indices = [118, 286, 302, 323, 460, 775, 941, 959, 978, 1114, 1436, 1454, 1627, 1767, 1784, 2147, 2163, 2331, 2348, 2489, 2839, 3008, 3025, 3043, 3182, 3516, 3534, 3704, 3720, 3858, 4200, 4366, 4387, 4403, 4540, 4889, 5058, 5080, 5217, 5234, 5589, 5759, 5778, 5796, 5933, 6270, 6438, 6456, 6473, 6611, 6868, 7036, 7053, 7070, 7207, 7487, 7504, 7671, 7688, 7832, 8162, 8334, 8351, 8489, 8506, 8814, 8831, 9002, 9017, 9157, 9492, 9510, 9682, 9700, 9836, 10178, 10348, 10365, 10504, 10519, 10859, 11025, 11046, 11064, 11202, 11531, 11548, 11717, 11735, 11874, 12244, 12261, 12430, 12572, 12590, 12931, 13102, 13122, 13260, 13277]

# result from run with MIN_SLICE_DISTANCE=10, NUM_TRIES=10000:
# ratio: [ 1.26905099  0.94491447  0.93540664  1.15758616]
# loss: 0.0716473201523
# test_indices_scattering = [3, 24, 38, 58, 75, 87, 103, 116, 136, 154, 167, 179, 195, 215, 230, 245, 258, 276, 297, 314, 326, 344, 362, 377, 394, 409, 426, 442, 456, 473, 489, 507, 521, 537, 552, 567, 585, 601, 618, 635, 647, 664, 683, 697, 713, 725, 741, 757, 774, 795, 808, 825, 841, 858, 876, 881, 898, 914, 933, 950, 967, 983, 997, 1010, 1029, 1048, 1063, 1080, 1098, 1115, 1124, 1137, 1153, 1170, 1186, 1212, 1227, 1242, 1261, 1274, 1287, 1300, 1318, 1338, 1353, 1367, 1383, 1400, 1418, 1436, 1448, 1464, 1479, 1494, 1510, 1525, 1542, 1560, 1582, 1594]
# test_indices = [118, 289, 303, 323, 460, 776, 942, 955, 975, 1113, 1443, 1455, 1621, 1641, 1776, 2150, 2163, 2331, 2352, 2489, 2843, 3011, 3029, 3044, 3181, 3524, 3691, 3707, 3721, 3858, 4204, 4372, 4386, 4402, 4537, 4894, 5062, 5078, 5095, 5232, 5595, 5762, 5781, 5795, 5931, 6272, 6438, 6454, 6471, 6612, 6871, 7038, 7054, 7071, 7209, 7487, 7504, 7670, 7689, 7826, 8165, 8331, 8345, 8358, 8497, 8820, 8985, 9002, 9020, 9157, 9495, 9508, 9674, 9691, 9827, 10186, 10351, 10366, 10505, 10518, 10862, 11025, 11043, 11063, 11198, 11536, 11702, 11719, 11737, 11875, 12250, 12416, 12431, 12446, 12582, 12935, 13102, 13120, 13262, 13274]

# result from run with MIN_SLICE_DISTANCE=3, NUM_TRIES=10000:
# ratio: [ 1.17793001  1.11320581  1.06825925  1.21175408]
# loss: 0.0677791041738
# test_indices_scattering = [11, 22, 40, 55, 71, 90, 103, 118, 128, 146, 171, 186, 204, 215, 229, 254, 272, 284, 296, 311, 332, 349, 360, 376, 392, 416, 428, 443, 452, 465, 491, 499, 512, 534, 547, 573, 587, 603, 617, 628, 648, 665, 679, 691, 710, 735, 752, 765, 778, 791, 807, 825, 846, 856, 871, 891, 910, 928, 937, 950, 974, 989, 1007, 1020, 1033, 1046, 1056, 1069, 1089, 1102, 1124, 1139, 1157, 1170, 1188, 1208, 1223, 1239, 1251, 1272, 1289, 1299, 1314, 1333, 1346, 1372, 1384, 1396, 1415, 1426, 1451, 1459, 1473, 1491, 1508, 1534, 1554, 1568, 1580, 1591]
# test_indices = [126, 287, 305, 320, 456, 779, 942, 957, 967, 1105, 1447, 1612, 1630, 1641, 1775, 2159, 2327, 2339, 2351, 2486, 2849, 3016, 3027, 3043, 3179, 3531, 3693, 3708, 3717, 3850, 4206, 4214, 4377, 4399, 4532, 4900, 5064, 5080, 5094, 5225, 5596, 5763, 5777, 5789, 5928, 6282, 6449, 6462, 6475, 6608, 6870, 7038, 7059, 7069, 7204, 7497, 7666, 7684, 7693, 7826, 8172, 8337, 8355, 8488, 8501, 8818, 8828, 8991, 9011, 9144, 9495, 9510, 9678, 9691, 9829, 10182, 10347, 10363, 10375, 10516, 10864, 10874, 11039, 11058, 11191, 11541, 11703, 11715, 11734, 11865, 12253, 12261, 12425, 12443, 12580, 12944, 13114, 13128, 13260, 13271]
