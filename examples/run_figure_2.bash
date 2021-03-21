#! /bin/bash
# Make sure you've activated the environment!

echo "Run figure 2 (iPINN, pristine specimen)"

# Case 0: Pristine specimen
# Root type: use known physics (wave equation i.e. this is an iPINN)
# loss type negative log probability
python hpm/train.py --data-type pristine --model-type ipinn
python hpm/post_process.py

echo "Results are found in \"results_params\" folder"
