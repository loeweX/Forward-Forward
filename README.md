# Reimplementation of the Forward-Forward Algorithm

This is a reimplementation of Geoffrey Hinton's Forward-Forward Algorithm in Python/Pytorch. The original
paper can be found [here](https://arxiv.org/abs/2212.13345) and the official implementation in
Matlab [here](https://www.cs.toronto.edu/~hinton/). Similarly to the Matlab implementation, this code covers the experiments described in section 3.3 ("A simple supervised example of FF") of the paper.

## Setup

Make sure you have conda installed (you can find instructions [here](https://www.anaconda.com/products/distribution)).

Then, run ```bash setup.sh``` to create a conda environment with all required packages.

The script installs PyTorch with CUDA 11.3. If you want to use a different version, you can change the version number in
the ```setup.sh``` script.

## Run Experiments

To train a model with forward-forward, run the following command:

```python -m main```

## Results

This reimplementation achieves slightly worse results than reported in the paper and 
roughly the same performance as what can be achieved with the official Matlab implementation. Here is the comparison:

| | Test Error |
| --- | -- |
| Paper | 1.36 |
| Matlab | 1.47 |
| This Repo | 1.45 |
