/**
 * @file mnist_loader.c
 * @brief Implementation of MNIST dataset loading functions
 */

#include "mnist_loader.h"
#include "neon_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Helper function to swap endianness (MNIST data is big-endian)
static uint32_t swap_endian(uint32_t val) {
    return ((val & 0xFF) << 24) | 
           ((val & 0xFF00) << 8) | 
           ((val & 0xFF0000) >> 8) | 
           ((val & 0xFF000000) >> 24);
}

bool read_idx_file(const char *filename, void *data, size_t size, uint32_t magic_expected) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open %s\n", filename);
        return false;
    }
    
    // Read magic number
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read magic number from %s\n", filename);
        fclose(file);
        return false;
    }
    
    magic = swap_endian(magic);
    if (magic != magic_expected) {
        fprintf(stderr, "Error: Invalid magic number in %s (expected 0x%08x, got 0x%08x)\n", 
                filename, magic_expected, magic);
        fclose(file);
        return false;
    }
    
    // Skip dimensions (we know the expected sizes)
    uint32_t dimensions[3];
    int dim_count = (magic_expected == 0x00000803) ? 3 : 1; // 3 dims for images, 1 for labels
    
    if (fread(dimensions, sizeof(uint32_t), dim_count, file) != (size_t)dim_count) {
        fprintf(stderr, "Error: Failed to read dimensions from %s\n", filename);
        fclose(file);
        return false;
    }
    
    // Read the data
    if (fread(data, 1, size, file) != size) {
        fprintf(stderr, "Error: Failed to read data from %s\n", filename);
        fclose(file);
        return false;
    }
    
    fclose(file);
    return true;
}

bool load_mnist(const char *base_path, MNISTData *data) {
    char path[512];
    
    // Default MNIST dataset dimensions
    data->num_train = MNIST_TRAIN_SIZE;
    data->num_test = MNIST_TEST_SIZE;
    data->image_size = MNIST_IMG_SIZE;
    data->num_classes = MNIST_NUM_CLASSES;
    
    // Allocate memory for MNIST data
    data->train_images = (float *)malloc(data->num_train * data->image_size * sizeof(float));
    data->train_labels = (int *)malloc(data->num_train * sizeof(int));
    data->test_images = (float *)malloc(data->num_test * data->image_size * sizeof(float));
    data->test_labels = (int *)malloc(data->num_test * sizeof(int));
    
    if (!data->train_images || !data->train_labels || !data->test_images || !data->test_labels) {
        fprintf(stderr, "Error: Failed to allocate memory for MNIST data\n");
        free_mnist_data(data);
        return false;
    }
    
    // Temporary buffers for raw data
    unsigned char *train_images_raw = (unsigned char *)malloc(data->num_train * data->image_size);
    unsigned char *test_images_raw = (unsigned char *)malloc(data->num_test * data->image_size);
    unsigned char *train_labels_raw = (unsigned char *)malloc(data->num_train);
    unsigned char *test_labels_raw = (unsigned char *)malloc(data->num_test);
    
    if (!train_images_raw || !test_images_raw || !train_labels_raw || !test_labels_raw) {
        fprintf(stderr, "Error: Failed to allocate memory for raw MNIST data\n");
        free(train_images_raw);
        free(test_images_raw);
        free(train_labels_raw);
        free(test_labels_raw);
        free_mnist_data(data);
        return false;
    }
    
    // Load training images
    sprintf(path, "%s/train-images-idx3-ubyte", base_path);
    if (!read_idx_file(path, train_images_raw, data->num_train * data->image_size, 0x00000803)) {
        free(train_images_raw);
        free(test_images_raw);
        free(train_labels_raw);
        free(test_labels_raw);
        free_mnist_data(data);
        return false;
    }
    
    // Load training labels
    sprintf(path, "%s/train-labels-idx1-ubyte", base_path);
    if (!read_idx_file(path, train_labels_raw, data->num_train, 0x00000801)) {
        free(train_images_raw);
        free(test_images_raw);
        free(train_labels_raw);
        free(test_labels_raw);
        free_mnist_data(data);
        return false;
    }
    
    // Load test images
    sprintf(path, "%s/t10k-images-idx3-ubyte", base_path);
    if (!read_idx_file(path, test_images_raw, data->num_test * data->image_size, 0x00000803)) {
        free(train_images_raw);
        free(test_images_raw);
        free(train_labels_raw);
        free(test_labels_raw);
        free_mnist_data(data);
        return false;
    }
    
    // Load test labels
    sprintf(path, "%s/t10k-labels-idx1-ubyte", base_path);
    if (!read_idx_file(path, test_labels_raw, data->num_test, 0x00000801)) {
        free(train_images_raw);
        free(test_images_raw);
        free(train_labels_raw);
        free(test_labels_raw);
        free_mnist_data(data);
        return false;
    }
    
    // Convert raw data to float/int
    for (int i = 0; i < data->num_train * data->image_size; i++) {
        data->train_images[i] = train_images_raw[i] / 255.0f;
    }
    
    for (int i = 0; i < data->num_test * data->image_size; i++) {
        data->test_images[i] = test_images_raw[i] / 255.0f;
    }
    
    for (int i = 0; i < data->num_train; i++) {
        data->train_labels[i] = train_labels_raw[i];
    }
    
    for (int i = 0; i < data->num_test; i++) {
        data->test_labels[i] = test_labels_raw[i];
    }
    
    // Free raw data buffers
    free(train_images_raw);
    free(test_images_raw);
    free(train_labels_raw);
    free(test_labels_raw);
    
    // Apply preprocessing
    preprocess_mnist_images(data->train_images, data->num_train, data->image_size, true);
    preprocess_mnist_images(data->test_images, data->num_test, data->image_size, true);
    
    return true;
}

void free_mnist_data(MNISTData *data) {
    if (data) {
        if (data->train_images) free(data->train_images);
        if (data->train_labels) free(data->train_labels);
        if (data->test_images) free(data->test_images);
        if (data->test_labels) free(data->test_labels);
        
        data->train_images = NULL;
        data->train_labels = NULL;
        data->test_images = NULL;
        data->test_labels = NULL;
    }
}

void preprocess_mnist_images(float *images, int num_images, int image_size, bool use_neon) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (use_neon) {
        // Neon-optimized preprocessing
        for (int i = 0; i < num_images; i++) {
            float *img = &images[i * image_size];
            
            // Step 1: Calculate mean
            float mean = 0.0f;
            for (int j = 0; j < image_size; j++) {
                mean += img[j];
            }
            mean /= image_size;
            
            // Step 2: Center the data
            for (int j = 0; j < image_size; j++) {
                img[j] -= mean;
            }
            
            // Step 3: Calculate standard deviation
            float variance = 0.0f;
            for (int j = 0; j < image_size; j++) {
                variance += img[j] * img[j];
            }
            variance /= image_size;
            float std_dev = sqrtf(variance + 1e-8f); // Add epsilon for numerical stability
            
            // Step 4: Normalize
            for (int j = 0; j < image_size; j++) {
                img[j] /= std_dev;
            }
            
            // Step 5: Scale to [0, 1] range
            float min_val = img[0], max_val = img[0];
            for (int j = 1; j < image_size; j++) {
                if (img[j] < min_val) min_val = img[j];
                if (img[j] > max_val) max_val = img[j];
            }
            float range = max_val - min_val;
            if (range > 1e-8f) {
                for (int j = 0; j < image_size; j++) {
                    img[j] = (img[j] - min_val) / range;
                }
            }
        }
    } else {
#endif
        // Standard preprocessing
        for (int i = 0; i < num_images; i++) {
            float *img = &images[i * image_size];
            
            // Calculate mean and standard deviation
            float mean = 0.0f, variance = 0.0f;
            for (int j = 0; j < image_size; j++) {
                mean += img[j];
            }
            mean /= image_size;
            
            for (int j = 0; j < image_size; j++) {
                float diff = img[j] - mean;
                variance += diff * diff;
            }
            variance /= image_size;
            float std_dev = sqrtf(variance + 1e-8f);
            
            // Normalize
            for (int j = 0; j < image_size; j++) {
                img[j] = (img[j] - mean) / std_dev;
            }
            
            // Rescale to [0, 1]
            float min_val = img[0], max_val = img[0];
            for (int j = 1; j < image_size; j++) {
                if (img[j] < min_val) min_val = img[j];
                if (img[j] > max_val) max_val = img[j];
            }
            float range = max_val - min_val;
            if (range > 1e-8f) {
                for (int j = 0; j < image_size; j++) {
                    img[j] = (img[j] - min_val) / range;
                }
            }
        }
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    }
#endif
}

void create_mnist_batch(const MNISTData *data, const int *batch_indices, int batch_size, 
                       float **batch_images, int *batch_labels) {
    for (int i = 0; i < batch_size; i++) {
        int idx = batch_indices[i];
        memcpy(batch_images[i], &data->train_images[idx * data->image_size], 
               data->image_size * sizeof(float));
        batch_labels[i] = data->train_labels[idx];
    }
}

void display_mnist_digit(const float *image, int label) {
    printf("Digit: %d\n", label);
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            float pixel = image[i * 28 + j];
            if (pixel < 0.2f) {
                printf("  ");
            } else if (pixel < 0.5f) {
                printf("· ");
            } else if (pixel < 0.8f) {
                printf("o ");
            } else {
                printf("@ ");
            }
        }
        printf("\n");
    }
}

void print_mnist_summary(const MNISTData *data) {
    printf("MNIST Dataset Summary\n");
    printf("---------------------\n");
    printf("Training samples: %d\n", data->num_train);
    printf("Test samples: %d\n", data->num_test);
    printf("Image size: %d pixels\n", data->image_size);
    printf("Number of classes: %d\n", data->num_classes);
    
    // Count samples per class in training set
    int train_counts[10] = {0};
    for (int i = 0; i < data->num_train; i++) {
        train_counts[data->train_labels[i]]++;
    }
    
    // Count samples per class in test set
    int test_counts[10] = {0};
    for (int i = 0; i < data->num_test; i++) {
        test_counts[data->test_labels[i]]++;
    }
    
    printf("\nClass distribution:\n");
    printf("Digit   Training    Test\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d     %5d     %4d\n", i, train_counts[i], test_counts[i]);
    }
    
    printf("\nSample digit:\n");
    display_mnist_digit(data->train_images, data->train_labels[0]);
}

void shuffle_indices(int *indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}
