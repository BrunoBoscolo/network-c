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
The project's source code is located in the `lib/` directory, with public headers in `include/`. The library is organized into the following modules:

-   **`gann`**: Provides the main high-level API (`gann.h`) for training and using networks.
-   **`neural_network`**: Contains the core logic for the neural network, including creation, forward propagation, and persistence.
-   **`matrix`**: A general-purpose matrix library for creating and manipulating the 2D matrices used for weights, biases, and data.
-   **`data_loader`**: Handles loading the MNIST dataset from its binary file format.
-   **`evolution`**: Implements the core evolutionary loop (`evo_create_initial_population`, `evo_reproduce`).
-   **`selection`**: Implements different parent selection strategies for the genetic algorithm (e.g., Tournament, Roulette Wheel).
-   **`crossover`**: Implements different crossover strategies for combining parent networks (e.g., Uniform, Single-Point).
-   **`mutation`**: Implements different mutation strategies for introducing genetic diversity (e.g., Gaussian, Uniform).
-   **`backpropagation`**: Contains the implementation of the backpropagation algorithm and its optimizers (SGD, Adam, RMSprop).
-   **`gann_errors`**: A simple, thread-safe error handling system.

## Getting Started

### Prerequisites
- A C compiler (e.g., `gcc` or `clang`)
- `make`
- (Optional) `doxygen` for generating documentation.
- (Optional) `libgtk-3-dev` for building the GUI example.

### Building the Project
The project uses a `Makefile` for building. The MNIST dataset is already included in the `data/` directory.

1.  **Build the example applications**:
    ```bash
    make all
    ```
    This will create several executables in the root directory, including `training` (for GA), `backprop_training` (for backprop), and `recognizer` (for evaluation).

### Running the Application

1.  **Train a new network with the Genetic Algorithm**:
    ```bash
    ./training
    ```
    This will train a new network and save the best one to `trained_network.dat`.

2.  **Train a new network with Backpropagation**:
    ```bash
    ./backprop_training
    ```

3.  **Run the Number Recognizer**:
    ```bash
    ./recognizer
    ```
    This will load the `trained_network.dat` file and evaluate its accuracy on the MNIST test set. You can also specify a different network file: `./recognizer my_network.dat`.

4.  **Run Other Examples**:
    The `examples/` directory contains several other executables for comparing genetic operators and activation functions. Use `make examples` to build them all.
    ```bash
    make examples
    ./examples/activations_comparison
    ./examples/comparison
    ```
    **Note:** The `recognizer_gui` example requires a graphical display and will not run in a headless environment.

### Running the Tests
The project includes a test suite using the `minunit` framework. To run the tests:
```bash
make test
```

## How It Works

A high-level API is provided in `gann.h` to make training easy. You only need to load your data, define the parameters, and call one of the main training functions.

### Reproducibility
For debugging or experiments, it's important to have reproducible results. This library uses a pseudo-random number generator for weight initialization and genetic operators. To ensure you get the same "random" results every time, seed the generator by calling `gann_seed_rng` at the beginning of your `main` function:

```c
#include "gann.h"
#include <time.h>

int main() {
    // Use a fixed seed for deterministic results during development
    gann_seed_rng(12345);

    // To get different results on each run, you can use the current time
    // gann_seed_rng(time(NULL));

    // ... your code here ...
}
```

### Training Methods
This library provides two different ways to train the neural network: a **Genetic Algorithm** and **Backpropagation**.

#### 1. Genetic Algorithm
The genetic algorithm is inspired by biological evolution. It works by evolving a population of networks over many generations.

1.  **Initialization**: An initial population of random neural networks is created.
2.  **Evaluation**: Each network is evaluated based on its performance on the training data. Its "fitness" is its accuracy.
3.  **Selection**: The top-performing networks ("parents") are selected for reproduction.
4.  **Reproduction**: The selected parents are combined using **crossover** to create new "child" networks. These children are then slightly changed with **mutation**.
5.  **Repeat**: This process repeats, and over time, the population evolves to become better at the task.

To train a network with the genetic algorithm, use the `gann_train` function. You can get a set of sensible default parameters by calling `gann_create_default_params()` and then overriding them as needed.

*Example (`examples/training.c`):*
```c
// Define the network architecture (input, hidden, output layers)
const int ARCHITECTURE[] = {MNIST_IMAGE_SIZE, 128, 64, MNIST_NUM_CLASSES};

// Get default training parameters
GannTrainParams params = gann_create_default_params();
params.architecture = ARCHITECTURE;
params.num_layers = sizeof(ARCHITECTURE) / sizeof(int);

// Start training
NeuralNetwork* best_net = gann_train(&params, train_dataset, NULL);
```

#### 2. Backpropagation
Backpropagation is a standard algorithm for training neural networks. It works by calculating the error of the network's predictions and then propagating this error backward through the network to adjust the weights and biases. This library supports three common optimization algorithms: **SGD**, **Adam**, and **RMSprop**.

To train a network with backpropagation, use the `gann_train_with_backprop` function.

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
NeuralNetwork* net = gann_train_with_backprop(&params, train_dataset, NULL);
```

### Advanced Usage: Custom Genetic Operators
For more advanced use cases, the `gann_evolve` function allows you to provide your own implementations for the core genetic operators. This is useful for experimenting with new selection, crossover, or mutation techniques.

You can define your own functions and pass them in a `GannEvolveParams` struct:
```c
// 1. Define your custom functions (examples)
NetworkFitness* my_selection(NetworkFitness* pop, int size, int* num_fittest, SelectionType type, int tour_size) { /* ... */ }
NeuralNetwork* my_crossover(const NeuralNetwork* p1, const NeuralNetwork* p2, CrossoverType type) { /* ... */ }
void my_mutation(NeuralNetwork* net, float rate, float chance, /*...*/) { /* ... */ }

// 2. Set up the parameters
GannEvolveParams evolve_params = {
    .base_params = gann_create_default_params(), // Start with defaults
    .selection_func = my_selection,
    .crossover_func = my_crossover,
    .mutation_func = my_mutation,
};
// ... set architecture, etc. on evolve_params.base_params ...

// 3. Start evolution
gann_evolve(&evolve_params, train_dataset, NULL);
```

## Documentation
The source code is documented using Doxygen-style comments. To generate a full HTML documentation set:

1.  **Install Doxygen**:
    ```bash
    # On Debian/Ubuntu
    sudo apt-get install doxygen
    # On macOS (using Homebrew)
    brew install doxygen
    ```
2.  **Generate Documentation**:
    ```bash
    make docs
    ```
This will create a `docs/` directory. Open `docs/html/index.html` in your browser to view the documentation.

## Contributing
Contributions are welcome. Please open an issue to discuss any changes.
