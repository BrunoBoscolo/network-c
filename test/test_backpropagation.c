#include "minunit.h"
#include "gann.h"
#include <math.h>

extern const double TEST_EPSILON;

// A simple test to see if the network can learn a single instance (overfit).
const char* test_backprop_overfit_single_instance() {
    // 1. Create a dummy dataset with one sample
    Dataset* dummy_dataset = create_dummy_dataset(1);
    mu_assert("Failed to create dummy dataset", dummy_dataset != NULL);

    // 2. Define network architecture and training parameters
    const int ARCHITECTURE[] = {dummy_dataset->images->cols, 10, dummy_dataset->labels->cols};
    GannBackpropParams params = {
        .architecture = ARCHITECTURE,
        .num_layers = sizeof(ARCHITECTURE) / sizeof(int),
        .learning_rate = 0.1,
        .epochs = 200, // More epochs to ensure overfitting
        .batch_size = 1,
        .activation_hidden = RELU,
        .activation_output = SIGMOID
    };

    // 3. Create and train the network
    NeuralNetwork* net = nn_create(params.num_layers, params.architecture, params.activation_hidden, params.activation_output);
    nn_init(net);
    backpropagate(net, dummy_dataset, &params);

    // 4. Test the prediction
    int prediction = gann_predict(net, dummy_dataset->images->data[0]);

    // Find the actual label from the one-hot encoded vector
    int actual_label = -1;
    for(int i=0; i < dummy_dataset->labels->cols; i++){
        if(fabs(dummy_dataset->labels->data[0][i] - 1.0) < TEST_EPSILON){
            actual_label = i;
            break;
        }
    }

    mu_assert("Prediction should match the label after training", prediction == actual_label);

    // 5. Cleanup
    nn_free(net);
    free_dataset(dummy_dataset);

    return NULL;
}
