#!/bin/bash
trap "exit" INT;
for m in learnedpd fbpmsdnet fbpunet fbpunetpp cinn ictnet fbpistaunet diptv tv fbp; do
    files=( /localdata/jleuschn/lodopab_reconstructions/${m}/* );
    if [ ${#files[@]} -eq 29 ]; then
        python lodopab_eval_saved.py --method $m --save_metrics;
    else
        echo skipping --method $m >&2
    fi;
done;
