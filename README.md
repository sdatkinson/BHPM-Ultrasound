# Bayesian Hidden Physics Models
Code accompanying Atkinson et al., [*Discovery of Physics and Characterization of Microstructure from Data with Bayesian Hidden Physics Models*](https://arxiv.org/abs/FIXME)

## Getting set up

### Data
Download the required data for the examples 
[here](https://drive.google.com/drive/folders/1e7w4yRblQFoPv_v6a4VibfNW1wVTKhDs?usp=sharing).
Unzip so that you have the following files relative to this README:
* `ultrasound_data/Hsample SAW 5MHz n2/wvf.mat`
* `ultrasound_data/30Jan15 Nist crack 240x240 12x12mm avg20 5MHz 0deg grips/wvf.mat`

### Configuring your environment
Experiments were run on a desktop running Ubuntu 18.04 with Python 3.7.3 with 
dependencies enumerated in `requirements-explicit.txt`.
A less verbose set of dependencies that should still work is given in 
`requirements.txt`.

Getting set up should be as simple as `pip install` with the right flags (i.e. `-r` and 
`-f` for JAX).
However, we've provided a convenience script; just do
```bash
. configure.bash
```
and you should be good.
Note that you may need to alter some parameters at the top of the script (e.g. CUDA 
version) to your specific setup.

## Running examples
A sequence of bash scripts are provided in `examples` to recreate the results reported in the paper.
Core functionality is packaged in the `bhpm` module.
