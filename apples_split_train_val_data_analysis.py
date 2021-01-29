import pandas as pd
import numpy as np

def make_apple_summary():
    data = pd.read_csv("supp_material/apples/train_slices_info.csv")
    pd.set_option('display.max_rows', 100)

    # list of apples
    apple_numbers = ['31101', '31102', '31103', '31104', '31105', '31106', '31107', '31111', '31112', '31113', '31114', '31115', '31116', '31118', '31119', '31121', '31201', '31202', '31203', '31204', '31205', '31206', '31207', '31209', '31211', '31213', '31214', '31215', '31216', '31217', '31218', '31219', '31220', '31221', '31301', '31302', '31303', '31304', '31305', '31306', '31307', '31309', '31310', '31311', '31312', '31313', '31316', '31317', '31318', '31319', '31320', '31321', '31322', '32101', '32102', '32104', '32106', '32107', '32108', '32111', '32112', '32113', '32114', '32116', '32118', '32119', '32120', '32121', '32122', '32201', '32202', '32203', '32205', '32206']

    data["Healthy slices"] = 1.
    data["Bitterpit slices"] = 0.
    data["Holes slices"] = 0.
    data["Rot slices"] = 0.
    data["Browning slices"] = 0.
    data.loc[data['bitterpit'] > 0., "Healthy slices"] = 0.0
    data.loc[data['bitterpit'] > 0., "Bitterpit slices"] = 1.0
    data.loc[data['holes'] > 0., "Healthy slices"] = 0.0
    data.loc[data['holes'] > 0., "Holes slices"] = 1.0
    data.loc[data['rot'] > 0., "Healthy slices"] = 0.0
    data.loc[data['rot'] > 0., "Rot slices"] = 1.0
    data.loc[data['browning'] > 0., "Healthy slices"] = 0.0
    data.loc[data['browning'] > 0., "Browning slices"] = 1.0
    data.loc[data['calix'] > 0., "Healthy slices"] = 0.0
    data.loc[data['stem'] > 0., "Healthy slices"] = 0.0
    data.loc[data['bruises'] > 0., "Healthy slices"] = 0.0

    # resulting table
    apple_df = pd.DataFrame(columns = data.columns)

    for num, apple_sample in enumerate(apple_numbers):
        select = data[data.apple_slice.str.contains(apple_sample)] # select all slices of an apple
        
        # select only a subset of slices
        limit_row = False
        if limit_row:
            mask = np.concatenate((
                        np.arange(193, 213, 1),
                        np.arange(363, 403, 1),
                        np.arange(523, 543, 1)
            ), axis = 0)
            select = select.iloc[mask]

        row_sum = select.sum(axis=0, skipna=True) # sum all columns
                
        apple_df = apple_df.append(row_sum, ignore_index=True)
        apple_df['apple_slice'][num] = apple_sample

    pixel_columns = ['bitterpit', 'holes', 'rot', 'calix', 'stem', 'bruises', 'core', 'browning', 'normal', 'background', 'unannotated']

    apple_df['Total pixels'] = 0.
    for col_name in pixel_columns:
        apple_df['Total pixels'] += apple_df[col_name]
        
    apple_df = apple_df.drop("calix", axis=1)
    apple_df = apple_df.drop("stem", axis=1)
    apple_df = apple_df.drop("bruises", axis=1)
    apple_df = apple_df.drop("core", axis=1)
    apple_df = apple_df.drop("normal", axis=1)
    apple_df = apple_df.drop("background", axis=1)
    apple_df = apple_df.drop("unannotated", axis=1)
    
    save_html = False
    if save_html:
        output = apple_df.to_html()
        with open("apple_df.html", "w") as text_file:
            text_file.write(output)
    
    return apple_df

def test_sequence(seq, apple_df):
    
    apple_nums = np.arange(0, 74)
    
    test_set = np.zeros((apple_nums.shape[0]), dtype = np.bool)
    test_set[seq] = True
    test_num = np.count_nonzero(test_set)
    test_stats = np.zeros((4))

    for sample in apple_nums[test_set]:
        test_stats[0] += apple_df.iloc[sample]["bitterpit"]
        test_stats[1] += apple_df.iloc[sample]["holes"]
        test_stats[2] += apple_df.iloc[sample]["rot"]
        test_stats[3] += apple_df.iloc[sample]["browning"]

    test_stats /= test_num
    return test_stats

def find_sequence(n_test, apple_df, tries=1000, verbose=True):
    
    apple_nums = np.arange(0, 74)
    
    test_set = np.zeros((apple_nums.shape[0]), dtype = np.bool)
    test_set[:n_test] = True

    defects_total = test_sequence(apple_nums, apple_df)
    
    rng = np.random.default_rng()
    best_seq = None
    best_loss = np.inf

    for i in range(tries):
        rng.shuffle(test_set)
        seq = np.nonzero(test_set)[0]
        defects = test_sequence(seq, apple_df)
        ratio = defects / defects_total
        loss = np.sum((1.-1./ratio)**2)
        if loss < best_loss:
            best_seq = seq
            best_loss = loss
            if verbose:
                print("New best test sequence: {}".format(seq))
                print("Relative average number of pixels for every defect: ")
                print("Bitterpit\tHoles\t\tRot\t\tBrowning")
                print("{:.3f}\t\t{:.3}\t\t{:.3f}\t{:.3f}".format(*ratio))
    return best_seq

def find_two_sequences(n_test1, n_test2, apple_df, tries=1000, verbose=True):
    
    apple_nums = np.arange(0, 74)
    
    assigned_set = np.zeros((apple_nums.shape[0]), dtype = np.int)
    assigned_set[:n_test1] = 1
    assigned_set[n_test1:n_test1+n_test2] = 2

    defects_total = test_sequence(apple_nums, apple_df)
    
    rng = np.random.default_rng()
    best_seqs = (None, None)
    best_loss = np.inf

    for i in range(tries):
        rng.shuffle(assigned_set)
        seq1 = np.nonzero(assigned_set == 1)[0]
        seq2 = np.nonzero(assigned_set == 2)[0]
        defects1 = test_sequence(seq1, apple_df)
        defects2 = test_sequence(seq2, apple_df)
        ratio1 = defects1 / defects_total
        ratio2 = defects2 / defects_total
        loss = np.sum((1.-1./ratio1)**2) + np.sum((1.-1./ratio2)**2)
        if loss < best_loss:
            best_seqs = (seq1, seq2)
            best_loss = loss
            if verbose:
                print("New best test sequences: {}, {}".format(seq1, seq2))
                print("Relative average number of pixels for every defect: ")
                print("Bitterpit\tHoles\t\tRot\t\tBrowning")
                print("{:.3f}\t\t{:.3}\t\t{:.3f}\t{:.3f}".format(*ratio1))
                print("{:.3f}\t\t{:.3}\t\t{:.3f}\t{:.3f}".format(*ratio2))
                print("loss: {}".format(loss))
    return best_seqs

if __name__ == "__main__":
    apple_df = make_apple_summary()
    print(apple_df)
    
    seq = np.arange(0, 74)
    print("Testing sequence {}".format(seq))
    defect_stats = test_sequence(seq, apple_df)
    print("Average number of pixels for every defect: ")
    print("Bitterpit\tHoles\t\tRot\t\tBrowning")
    print("{:.3E}\t{:.3E}\t{:.3E}\t{:.3E}".format(*defect_stats))

    # find one sequence with similar defect stats
    print("")
    test_seq = find_sequence(8, apple_df, tries=100000)
    print("")
    print("Best test sequence found: {}".format(test_seq))
    test_defect_stats = test_sequence(test_seq, apple_df)
    print("Average number of pixels for every defect: ")
    print("Bitterpit\tHoles\t\tRot\t\tBrowning")
    print("{:.3E}\t{:.3E}\t{:.3E}\t{:.3E}".format(*test_defect_stats))

    # # find two sequences with similar defect stats
    # print("")
    # test_seqs = find_two_sequences(7, 7, apple_df, tries=100000)
    # print("")
    # print("Best test sequences found: {}".format(test_seqs))
    # test_defect_stats1 = test_sequence(test_seqs[0], apple_df)
    # test_defect_stats2 = test_sequence(test_seqs[1], apple_df)
    # print("Average number of pixels for every defect: ")
    # print("Bitterpit\tHoles\t\tRot\t\tBrowning")
    # print("{:.3E}\t{:.3E}\t{:.3E}\t{:.3E}".format(*test_defect_stats1))
    # print("{:.3E}\t{:.3E}\t{:.3E}\t{:.3E}".format(*test_defect_stats2))


# Example result for one test sequence:
# [11, 12, 16, 33, 36, 42, 55, 61]

# Example result for two test sequences:
# [3, 16, 30, 39, 57, 66, 71], [26, 42, 44, 48, 53, 55, 69]
