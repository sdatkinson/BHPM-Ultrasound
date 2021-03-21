#! /bin/bash
# Make sure you've activated the environment!

echo "Run figure 6 (BHPM, cracked specimen)"
echo "****************NOTE*****************"
echo "You MUST either run figure 5 right before this"
echo "or change the f_from arg to the timestamp of the Fig 5 run that you want!"

# * 200k iters on u required to geet better fit.
# * f-from: get physics operator f() from Fig 5 run (assumed to be the latest previous 
#   run)
python hpm/train.py \
--data-type cracked \
--model-type bhpm \
--u-iters 200000 \
--f-from latest

python hpm/post_process.py

echo "Results are found in \"results_params\" folder"
