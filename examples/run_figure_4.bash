#! /bin/bash
# Make sure you've activated the environment!

echo "Run figure 4 (iPINN, cracked specimen)"

# Case 3: Crack, "hard mode"
# Root type: use known physics (wave equation i.e. this is an iPINN)
# loss type negative log probability
# Need mroe iters to capture solution
python hpm/train.py --data-type cracked --model-type ipinn --u-iters 200000
python hpm/post_process.py

echo "Results are found in \"results_params\" folder"
