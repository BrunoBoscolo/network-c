#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural_network.h"
#include "data_loader.h"

// Helper function to get the predicted class from the network's output
int get_predicted_class(const Matrix* output) {
    int max_index = 0;
    for (int i = 1; i < output->cols; i++) {
        if (output->data[0][i] > output->data[0][max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// Helper function to get the true class from the one-hot encoded label
int get_true_class(const double* label_row) {
    for (int i = 0; i < MNIST_NUM_CLASSES; i++) {
        if (label_row[i] == 1.0) {
            return i;
        }
    }
    return -1; // Should not happen
}

int main(int argc, char* argv[]) {
    printf("--- MNIST Number Recognizer ---\n");

    const char* network_filepath = "trained_network.dat";
    if (argc > 1) {
        network_filepath = argv[1];
        printf("Loading network from specified file: %s\n", network_filepath);
    } else {
        printf("Loading network from default file: %s\n", network_filepath);
    }

    // 1. Load the pre-trained network
    NeuralNetwork* net = load_network(network_filepath);
    if (!net) {
        fprintf(stderr, "Failed to load network from %s. Please train a model first by running './main'.\n", network_filepath);
        return 1;
    }
    printf("Network loaded successfully.\n");
    printf("Network architecture: [");
    for(int i=0; i<net->num_layers; i++) {
        printf("%d%s", net->architecture[i], i == net->num_layers - 1 ? "" : ", ");
    }
    printf("]\n");

    // 2. Load the MNIST test dataset
    printf("Loading MNIST test data...\n");
    Dataset* test_dataset = load_mnist_dataset("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
    if (!test_dataset) {
        fprintf(stderr, "Failed to load the MNIST test dataset.\n");
        free_neural_network(net);
        return 1;
    }
    printf("Test data loaded: %d images.\n", test_dataset->num_items);

    // 3. Evaluate the network on the test dataset
    printf("Evaluating network accuracy...\n");
    int correct_predictions = 0;
    for (int i = 0; i < test_dataset->num_items; i++) {
        Matrix* input = create_matrix(1, MNIST_IMAGE_SIZE);
        if (!input) continue;
        memcpy(input->data[0], test_dataset->images->data[i], MNIST_IMAGE_SIZE * sizeof(double));

        Matrix* output = forward_pass(net, input);
        if (!output) {
            free_matrix(input);
            continue;
        }

        int predicted_class = get_predicted_class(output);
        int true_class = get_true_class(test_dataset->labels->data[i]);

        if (predicted_class == true_class) {
            correct_predictions++;
        }

        free_matrix(input);
        free_matrix(output);
    }

    // 4. Calculate and print the final accuracy
    double accuracy = (double)correct_predictions / test_dataset->num_items;
    printf("----------------------------------\n");
    printf("Final Accuracy on Test Set: %.2f%% (%d/%d correct)\n",
           accuracy * 100.0, correct_predictions, test_dataset->num_items);
    printf("----------------------------------\n");

    // 5. Cleanup
    free_neural_network(net);
    free_dataset(test_dataset);

    return 0;
}
