# Neural Network Library in C

This project is a C implementation of a simple feedforward neural network that can be trained with either a genetic algorithm or backpropagation. It is designed to be a learning tool for beginners and is pre-configured to solve the MNIST handwritten digit recognition problem.

## Features
- **Feedforward Neural Network**: A simple, fully connected neural network implementation from scratch in C.
- **Two Training Methods**:
    - **Genetic Algorithm**: Evolve a population of networks to solve a problem. Includes multiple selection, crossover, and mutation methods.
    - **Backpropagation**: Train a network with gradient descent. Includes classic optimizers like **SGD**, **Adam**, and **RMSprop**.
- **MNIST Dataset**: The project is pre-configured to work with the MNIST dataset of handwritten digits.
- **Configurable Activation Functions**: Supports Sigmoid, ReLU, and Leaky ReLU for hidden layers.
- **Modular Architecture**: The code is organized into separate modules for the neural network, training algorithms, data loading, and matrix operations.
- **Build and Test with Make**: A `Makefile` is provided for easy building and testing of the project.
- **Network Persistence**: The trained network can be saved to a file and loaded later for evaluation.
- **Reproducible Results**: The random number generator can be seeded to ensure that training is deterministic.

## Architecture
The project's source code is located in the `lib/` directory and is organized into four main components:
- `matrix`: A general-purpose matrix library for creating and manipulating 2D matrices.
- `neural_network`: Contains the core logic for the neural network, including network creation, forward propagation, mutation, and persistence.
- `evolution`: Implements the genetic algorithm, including population creation, fitness evaluation, selection, crossover, and reproduction.
- `data_loader`: Handles loading the MNIST dataset from files into a format that can be used by the neural network.

## Getting Started

### Prerequisites
- A C compiler (e.g., `gcc`)
- `make`

### Building the Project
The project uses a `Makefile` for building. The MNIST dataset is already included in the `data/` directory.

1.  **Build the training and recognition applications**:
    ```bash
    make all
    make recognizer
    ```
    This will create two executables: `main` for training, and `recognizer` for evaluating a trained network.

### Running the Application

1.  **Train a new network**:
    ```bash
    ./main
    ```
    This will train a new network and save the best one to `trained_network.dat`.

2.  **Run the Number Recognizer**:
    ```bash
    ./recognizer
    ```
    This will load the `trained_network.dat` file and evaluate its accuracy on the MNIST test set. You can also specify a different network file as a command-line argument:
    ```bash
    ./recognizer my_network.dat
    ```

3.  **Compare Activation Functions**:
    A separate example is provided to compare the performance of the different activation functions.
    ```bash
    ./examples/activations_comparison
    ```
    This will train a network for each activation function (Sigmoid, ReLU, Leaky ReLU) and print the final accuracy of each.

### Running the Tests
The project includes a test suite to verify the correctness of the core components. To run the tests, use the following command:
```bash
make test
```

## How It Works

A high-level API is provided in `gann.h` to make training easy. You only need to load your data, define the parameters, and call one of the training functions.

### Reproducibility
For debugging or scientific experiments, it's important to have reproducible results. This library uses a pseudo-random number generator for initializing network weights and for some genetic operators. To ensure that you get the same "random" results every time you run the program, you can seed the generator.

To do this, call the `gann_seed_rng` function at the beginning of your `main` function:
```c
#include "gann.h"
#include <time.h>

int main() {
    // Use a fixed seed for deterministic results
    gann_seed_rng(12345);

    // To get different results on each run, you can use the current time as a seed
    // gann_seed_rng(time(NULL));

    // ... your code here ...
}
```

### Neural Network
The neural network is a simple feedforward network. It takes a flattened 28x28 (784-pixel) image as input and passes it through a series of layers. The output layer has 10 neurons, one for each digit (0-9). The neuron with the highest activation is the network's guess.

### Training Methods
This library provides two different ways to train the neural network: a **Genetic Algorithm** and **Backpropagation**.

#### 1. Genetic Algorithm
The genetic algorithm is inspired by biological evolution. It's a great way to learn about how evolution can be used as an optimization technique.
1.  **Initialization**: An initial population of random neural networks is created.
2.  **Evaluation**: Each network in the population is evaluated based on its performance on the MNIST dataset. Its "fitness" is the number of digits it correctly identifies.
3.  **Selection**: The top-performing networks (the "fittest") are selected to be "parents" for the next generation.
4.  **Reproduction**: The selected parents are combined using crossover to create new "child" networks. These children are then slightly mutated. These new networks form the next generation.
5.  **Repeat**: The process is repeated for many generations, and over time, the population of networks evolves to become better at recognizing digits.

To train a network with the genetic algorithm, use the `gann_train` function. You can get a set of sensible default parameters by calling `gann_create_default_params()` and then override them as needed.

*Example (`examples/training.c`):*
```c
// Define the network architecture (input, hidden, output layers)
const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};

// Get default training parameters
GannTrainParams params = gann_create_default_params();
params.architecture = ARCHITECTURE;
params.num_layers = sizeof(ARCHITECTURE) / sizeof(int);

// Start training
NeuralNetwork* best_net = gann_train(&params, train_dataset);
```

#### 2. Backpropagation
Backpropagation is a standard algorithm for training neural networks. It works by calculating the error (or "loss") of the network's predictions and then propagating this error backward through the network to adjust the weights and biases. This library supports three common optimization algorithms: **SGD**, **Adam**, and **RMSprop**.

To train a network with backpropagation, use the `gann_train_with_backprop` function. You need to fill out the `GannBackpropParams` struct with your desired configuration.

*Example (`examples/backprop_training.c`):*
```c
// Define the network architecture
const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};

// Define backpropagation parameters
GannBackpropParams params = {
    .architecture = ARCHITECTURE,
    .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
    .learning_rate = 0.001,
    .epochs = 5,
    .batch_size = 32,
    .optimizer_type = ADAM, // Choose between SGD, ADAM, RMSPROP
};

// Start training
NeuralNetwork* net = gann_train_with_backprop(&params, train_dataset);
```

## Contributing
Contributions are welcome. Please open an issue to discuss any changes.
