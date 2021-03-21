#! /bin/bash
# Make sure you've activated the environment!

echo "Run figure 5 (BHPM, pristine specimen)"

# Note: More than 20k u-iters and 50k af-iters would be great, but this is what was run 
# at the time of doing the experiments for the paper and is what's needed to replicate 
# the results exactly.
python hpm/train.py \
--data-type pristine \
--model-type bhpm \
--u-iters 20000 \
--af-iters 40000
python hpm/post_process.py

echo "Results are found in \"results_params\" folder"
