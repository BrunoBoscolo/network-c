#include <stdio.h>
#include <stdlib.h>
#include "gann.h"

int main(int argc, char* argv[]) {
    printf("--- MNIST Number Recognizer (Simple API) ---\n");

    const char* network_filepath = "trained_network.dat";
    if (argc > 1) {
        network_filepath = argv[1];
        printf("Loading network from specified file: %s\n", network_filepath);
    } else {
        printf("Loading network from default file: %s\n", network_filepath);
    }

    // 1. Load the pre-trained network
    NeuralNetwork* net = nn_load(network_filepath);
    if (!net) {
        fprintf(stderr, "Failed to load network. Please run the training example first.\n");
        return 1;
    }

    // 2. Load the MNIST test dataset
    Dataset* test_dataset = load_mnist_dataset("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    if (!test_dataset) {
        fprintf(stderr, "Failed to load the MNIST test dataset.\n");
        nn_free(net);
        return 1;
    }

    // 3. Evaluate the network on the test dataset using the simple API
    printf("Evaluating network accuracy...\n");
    double accuracy = gann_evaluate(net, test_dataset);
    int correct_predictions = (int)(accuracy * test_dataset->num_items);

    // 4. Print the final accuracy
    printf("----------------------------------\n");
    printf("Final Accuracy on Test Set: %.2f%% (%d/%d correct)\n",
           accuracy * 100.0, correct_predictions, test_dataset->num_items);
    printf("----------------------------------\n");

    // 5. Cleanup
    nn_free(net);
    free_dataset(test_dataset);

    return 0;
}
