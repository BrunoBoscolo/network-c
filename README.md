# Neural Network Evolution in C

This project is a C implementation of a simple feedforward neural network that is trained using a genetic algorithm. It is designed to solve the MNIST handwritten digit recognition problem.

## Features
- **Feedforward Neural Network**: A simple, fully connected neural network implementation from scratch in C.
- **Genetic Algorithm**: The network is trained using a genetic algorithm, where a population of networks "evolves" to become better at recognizing digits.
- **Crossover**: The genetic algorithm now uses crossover to create new networks from the fittest parents.
- **MNIST Dataset**: The project is pre-configured to work with the MNIST dataset of handwritten digits.
- **Modular Architecture**: The code is organized into separate modules for the neural network, the genetic algorithm, and data loading.
- **Build and Test with Make**: A `Makefile` is provided for easy building and testing of the project.
- **Network Persistence**: The trained network can be saved to a file and loaded later for evaluation.

## Architecture
The project is divided into three main components:
- `neural_network`: Contains the core logic for the neural network, including matrix operations, network creation, forward propagation, mutation, and persistence.
- `evolution`: Implements the genetic algorithm, including population creation, fitness evaluation, selection, crossover, and reproduction.
- `data_loader`: Handles loading the MNIST dataset from files into a format that can be used by the neural network.

## Getting Started

### Prerequisites
- A C compiler (e.g., `gcc`)
- `make`

### Building the Project
The project uses a `Makefile` for building.

1.  **Download the MNIST dataset**:
    A shell script is provided to download the necessary data.
    ```bash
    ./download_mnist.sh
    ```

2.  **Build the training and recognition applications**:
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

### Running the Tests
The project includes a test suite to verify the correctness of the core components. To run the tests, use the following command:
```bash
make test
```

## How It Works
The project combines two main concepts: neural networks and genetic algorithms.

### Neural Network
The neural network is a simple feedforward network. It takes a flattened 28x28 (784-pixel) image as input and passes it through a series of layers. The output layer has 10 neurons, one for each digit (0-9). The neuron with the highest activation is the network's guess.

### Genetic Algorithm
Instead of using a traditional training algorithm like backpropagation, this project uses a genetic algorithm:
1.  **Initialization**: An initial population of random neural networks is created.
2.  **Evaluation**: Each network in the population is evaluated based on its performance on the MNIST dataset. Its "fitness" is the number of digits it correctly identifies.
3.  **Selection**: The top-performing networks (the "fittest") are selected to be "parents" for the next generation.
4.  **Reproduction**: The selected parents are combined using crossover to create new "child" networks. These children are then slightly mutated. These new networks form the next generation.
5.  **Repeat**: The process is repeated for many generations, and over time, the population of networks evolves to become better at recognizing digits.

## Contributing
Contributions are welcome. Please open an issue to discuss any changes.
