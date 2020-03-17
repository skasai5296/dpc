# DPC
Dense Predictive Coding
---

Code is still in progress.


## Requirements (libraries)
```
python>=3.7
numpy
matplotlib
pytorch>=1.3.0
torchvision>=0.4.0
Pillow
addict
pyyaml
h5py (for converting .jpg files to .hdf5)
wandb
ffmpeg (for converting .mp4 files to .jpg)
joblib (for parallel conversions of .mp4 files to .jpg)
```


### For Anaconda Users
- `environment.yml` file contains environment details for Anaconda users.
- run `conda env create -f environment.yml && conda activate dpc` for simple use.

## Preparation of Dataset
- Download Kinetics 700 mp4 files
- Prepare HDF5 files for Kinetics 700 using `src/util/convert_mp4_hdf5.py`

## Training and validation
Run `python train.py` (by default, the system uses cfg/debug.yml as configuration for hyperparameters)
For custom configuration, use `python train.py --config ${PATH_TO_YOUR_CUSTOM_CONFIG}`

## Evaluation

## Description
This repository contains the implementation of dense predictive coding.
Training and Evaluation is done on the Kinetics 700 dataset.
