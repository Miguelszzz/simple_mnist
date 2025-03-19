/**
 * @file mnist_loader.h
 * @brief Functions for loading and preprocessing MNIST data
 */

#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// MNIST dataset constants
#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE 10000
#define MNIST_IMG_SIZE 784  // 28x28 pixels
#define MNIST_NUM_CLASSES 10

/**
 * @brief MNIST dataset structure
 */
typedef struct {
    float *train_images;   // Training images [60000 x 784]
    int *train_labels;     // Training labels [60000]
    float *test_images;    // Test images [10000 x 784]
    int *test_labels;      // Test labels [10000]
    int num_train;         // Number of training samples
    int num_test;          // Number of test samples
    int image_size;        // Size of each image (pixels)
    int num_classes;       // Number of classes
} MNISTData;

/**
 * @brief Load MNIST dataset from binary files
 * 
 * @param base_path Path to directory containing MNIST files
 * @param data Pointer to MNISTData structure to fill
 * @return true if loading was successful, false otherwise
 */
bool load_mnist(const char *base_path, MNISTData *data);

/**
 * @brief Free memory allocated for MNIST data
 * 
 * @param data Pointer to MNISTData structure
 */
void free_mnist_data(MNISTData *data);

/**
 * @brief Preprocess MNIST images (normalize, enhance contrast, etc.)
 * 
 * @param images Array of images
 * @param num_images Number of images
 * @param image_size Size of each image
 * @param use_neon Whether to use ARM Neon optimizations
 */
void preprocess_mnist_images(float *images, int num_images, int image_size, bool use_neon);

/**
 * @brief Create a mini-batch from MNIST training data
 * 
 * @param data MNIST data structure
 * @param batch_indices Array of indices for the batch
 * @param batch_size Size of the batch
 * @param batch_images Output array for batch images [batch_size x image_size]
 * @param batch_labels Output array for batch labels [batch_size]
 */
void create_mnist_batch(const MNISTData *data, const int *batch_indices, int batch_size, 
                       float **batch_images, int *batch_labels);

/**
 * @brief Helper function to display MNIST digit in ASCII art
 * 
 * @param image Pointer to image data (784 pixels)
 * @param label Label of the image
 */
void display_mnist_digit(const float *image, int label);

/**
 * @brief Helper function to read IDX file format
 * 
 * @param filename IDX file to read
 * @param data Pointer to buffer to store data (must be pre-allocated)
 * @param size Size of data to read
 * @param magic_expected Expected magic number
 * @return true if reading was successful, false otherwise
 */
bool read_idx_file(const char *filename, void *data, size_t size, uint32_t magic_expected);

/**
 * @brief Display summary statistics for MNIST dataset
 * 
 * @param data MNIST data structure
 */
void print_mnist_summary(const MNISTData *data);

/**
 * @brief Shuffle indices for batch creation
 * 
 * @param indices Array of indices to shuffle
 * @param n Number of indices
 */
void shuffle_indices(int *indices, int n);

#endif /* MNIST_LOADER_H */
