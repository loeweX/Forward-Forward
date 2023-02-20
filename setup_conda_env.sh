#!/bin/bash

# Create conda environment.
conda env create -f environment.yml
conda activate FF

# Install additional packages.
conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install hydra-core
pip install einops