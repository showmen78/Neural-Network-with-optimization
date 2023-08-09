
# Neural Network Implementation from Scratch

This repository contains a Python implementation of a feedforward Artificial Neural Network (ANN) built from scratch,   implemented for handwritten digit recognition. The implementation includes customizable layers, various activation functions, regularization options, and different optimization techniques.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This neural network implementation aims to provide a simple yet flexible framework for building and training feedforward neural networks. It includes the following key features:

- Customizable layers with options for different activation functions.
- Support for `binary classification` and `multiclass classification`.
- Regularization techniques such as `L2` and `L1` and `dropout` regularization.
- Optimization methods including `momentum`, `RMSprop`, and `Adam`.
- Learning rate scheduling through `learning rate decay`.

## Getting Started

To get started, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/showmen78/Neural-Network-with-optimization.git
   cd Neural-Network-with-optimization
   ```

2. Install the required dependencies. The code is built using Python and requires the following libraries: `numpy`,`matplotlib`,`math`. You can install them using:

   ```bash
   pip install numpy matplotlib
   ```

## Usage

To use the neural network implementation, you can follow these steps:

1. Import the required classes and functions:

   ```python
   from ann import Ann
   ```

2. Prepare your training and testing data. Make sure to normalize your data between 0 and 1. The input data should be a matrix of shape `(input_feature,no of samples)` and output data should be of shape `(no of output class, no of samples)`

3. Configure the hyperparameters and create an instance of the `Ann` class:

   ```python
   model = Ann(
       neurons=[10],  # List of neurons per layer
       activation=['softmax'],  # Activation functions for each layer
       reg='L2', lamd=10,  # Regularization options
       beta=0.85, optimizer='rms', lr_decay='exp_decay', decay_rate=0.2,
       _type='multi_classification'  # Choose classification type
   )
   ```

4. Train the model using the `train` method:

   ```python
   model.train(
       X_train, Y_train, batch_size=50,
       X_test=X_test, Y_test=Y_test,
       lr=0.2, epoch=50, show_acc=[True, True]
   )
   ```

5. Predict using the trained model:

   ```python
   accuracy, predictions = model.predict(X_test, Y_test)
   print(f"Test Accuracy: {accuracy}%")
   ```

## Customization

You can customize the neural network by modifying the following parameters:

- `neurons`: List of neurons per layer. eg [10,20].
- `activation`: List of activation functions for each layer. eg ['sigmoid','softmax'].
- `reg`: Regularization method ('L2', 'L1', 'dropout' or 'none').
- `lamd`: Lambda value for regularization.
- `beta`: Optimization parameter for momentum,RMSprop and Adam.
- `optimizer`: Optimization method ('momentum', 'rms', or 'adam').
- `lr_decay`: Learning rate decay method ('lr_decay', 'exp_decay', or 'none').
- `decay_rate`: Decay rate for learning rate decay.
- `_type`: Classification type ('binary_classification' or 'multi_classification').

## Contributing

Contributions to this repository are welcome! If you find any issues or have ideas for improvements, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

