# Reimplementation of the Forward-Forward Algorithm

This is a reimplementation of Geoffrey Hinton's Forward-Forward Algorithm in Python/Pytorch. The original
paper can be found [here](https://arxiv.org/abs/2212.13345) and the official implementation in
Matlab [here](https://www.cs.toronto.edu/~hinton/). Similarly to the Matlab implementation, this code covers the 
experiments described in section 3.3 ("A simple supervised example of FF") of the paper.

## The Forward-Forward Algorithm

The Forward-Forward algorithm is a method for training deep neural networks in a more biologically plausible manner.
Instead of sharing gradients between layers, it trains each layer based on local losses. 

To implement these local losses, the network performs two forward passes:
The first forward pass is on positive samples, which are representative of the "real" data. 
For these samples, the network is trained to maximize the "goodness" for each of its layers. 
In the second forward pass, the network is fed negative samples, 
which are data perturbations that do not conform to the true data distribution. 
For these samples, the network is trained to minimize the "goodness".

The goodness can be evaluated in several ways, such as by taking the sum of the squared activities of a layer.

<img src="ForwardForward.jpeg" alt="The Forward-Forward Algorithm" width="600"/>

The image above depicts the training of a network with the Forward-Forward algorithm as implemented in this repository. 
Here, the positive and negative samples are created by adding a one-hot encoding of the correct or incorrect label 
to the first ten pixels of the image.


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
