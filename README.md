# DPGNN-GasLeak (Deep Probabilistic Graph Neural Network for Natural Gas Leak Detection)

This repository contains code and data for replicating the work in the paper 'Towards deep probabilistic graph neural network for natural gas leak detection and localization without labeled anomaly data' by Zhang et al. (2023). The paper proposes a deep probabilistic graph neural network that integrates an attention-based graph neural network and variational Bayesian inference to model spatial sensor dependency and localize natural gas leaks.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

- Install [PyTorch](https://pytorch.org/) and any required dependencies on your development environment.

### Data Preprocessing

Preprocess the benchmark dataset as described in section 4.2 of the paper to normalize the time-series data.

### Model Implementation

1. Implement the attention-based graph neural network as described in section 3.1 of the paper using PyTorch’s built-in neural network modules and functions.
2. Implement the variational Bayesian inference component as described in section 3.2 of the paper using PyTorch’s probabilistic programming capabilities.
3. Integrate the attention-based graph neural network and variational Bayesian inference components to create the VB_GAnomaly model as described in section 3.3 of the paper.

### Training

Train the VB_GAnomaly model on the preprocessed benchmark dataset using PyTorch’s optimization and training capabilities.

### Evaluation

Evaluate the performance of the trained VB_GAnomaly model on a test dataset using the evaluation metrics described in section 3.4 of the paper.

## License

This project is licensed under the [GPL License](LICENSE).

