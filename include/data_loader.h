#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "neural_network.h"

/**
 * @file data_loader.h
 * @brief Functions for loading and managing datasets.
 * @details This file provides utilities for loading the MNIST dataset from its
 * standard file format and for managing the `Dataset` struct.
 */

#define MNIST_IMAGE_ROWS 28
#define MNIST_IMAGE_COLS 28
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS)
#define MNIST_NUM_CLASSES 10

/**
 * @brief Represents a dataset of images and corresponding labels.
 */
typedef struct {
    int num_items;  /**< The total number of items (image-label pairs) in the dataset. */
    Matrix* images; /**< A matrix where each row is a flattened image, normalized to values between 0.0 and 1.0. */
    Matrix* labels; /**< A matrix where each row is a one-hot encoded vector representing the label. */
} Dataset;

// --- Data Loader Functions ---

/**
 * @brief Loads the MNIST dataset from the specified IDX-formatted files.
 * @details This function reads the binary IDX files for both images and labels,
 * performs endian swapping for the header information, normalizes pixel values
 * to be between 0.0 and 1.0, and one-hot encodes the labels.
 * @param image_path The file path to the MNIST image data (e.g., "train-images.idx3-ubyte").
 * @param label_path The file path to the MNIST label data (e.g., "train-labels.idx1-ubyte").
 * @return A pointer to a new `Dataset` struct containing the loaded data.
 * @return `NULL` on failure (e.g., file not found, format error). The caller is
 *         responsible for freeing the returned dataset using `free_dataset()`.
 */
Dataset* load_mnist_dataset(const char* image_path, const char* label_path);

/**
 * @brief Creates a dummy dataset with random values for testing purposes.
 * @param num_items The number of items (images and labels) to create in the dataset.
 * @return A pointer to the created `Dataset`. The caller is responsible for
 *         freeing this dataset using `free_dataset()`.
 */
Dataset* create_dummy_dataset(int num_items);

/**
 * @brief Creates a dummy dataset with a specific label for all items.
 * @details This is useful for testing if a network can overfit to a single class.
 * @param num_items The number of items to create.
 * @param label The integer label (0-9) to assign to all items.
 * @return A pointer to the created `Dataset`. The caller is responsible for
 *         freeing this dataset using `free_dataset()`.
 */
Dataset* create_dummy_dataset_with_label(int num_items, int label);

/**
 * @brief Splits a dataset into two new datasets by copying the data.
 * @details This function is useful for creating a training and validation set from a
 * single source dataset. It creates two new datasets and deep copies the
 * corresponding data from the original.
 * @param original The source dataset to split.
 * @param split_size The number of items from the end of the original dataset to put in the second dataset (`out_dataset_2`).
 * @param out_dataset_1 A pointer to a `Dataset` struct that will be populated with the first part of the split.
 * @param out_dataset_2 A pointer to a `Dataset` struct that will be populated with the second part of the split.
 */
void split_dataset(const Dataset* original, int split_size, Dataset* out_dataset_1, Dataset* out_dataset_2);

/**
 * @brief Frees the memory allocated for a dataset.
 * @details Deallocates the `images` matrix, `labels` matrix, and the `Dataset` struct itself.
 * It is safe to pass `NULL` to this function.
 * @param dataset The dataset to free.
 */
void free_dataset(Dataset* dataset);

#endif // DATA_LOADER_H
