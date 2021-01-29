#!/bin/bash
trap "exit" INT;
for n in noisefree gaussian_noise scattering; do
    if [ $n == "gaussian_noise" ]; then
        noise_folder_name="gaussian";
    else
        noise_folder_name=$n;
    fi;
    for a in 2 5 10 50; do
        for m in learnedpd fbpmsdnet fbpunet cinn ictnet fbpistaunet tv cgls fbp; do
            files=( /localdata/jleuschn/apples_reconstructions/${m}/${noise_folder_name}/ang${a}/* );
            if [ ${#files[@]} -eq 100 ]; then
                python apples_eval_saved_data_discrepancy.py --method $m --noise_setting $n --num_angles $a --secret_test_set --save_data_discrepancy;
            else
                echo skipping --method $m --noise_setting $n --num_angles $a >&2
            fi;
        done;
    done;
done;
