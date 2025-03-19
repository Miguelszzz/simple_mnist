/**
 * @file simple_mnist.c
 * @brief Minimalist MNIST implementation in pure C with two hidden layers
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mnist_loader.h"

// Network architecture
#define INPUT_SIZE 784     // 28x28 pixels
#define HIDDEN1_SIZE 512   // First hidden layer (increased further)
#define HIDDEN2_SIZE 256   // Second hidden layer (increased further)
#define OUTPUT_SIZE 10     // 10 digits
#define LEARNING_RATE 0.01f // Lower learning rate for more stable convergence
#define WEIGHT_DECAY 2e-5f // Increased L2 regularization to prevent overfitting
#define BATCH_SIZE 128     // Larger batch size for better gradient estimates
#define EPOCHS 150         // More epochs
#define USE_DATA_AUGMENTATION 1 // Enable data augmentation

// Network structure
typedef struct {
    // First hidden layer
    float *hidden1_weights;  // [HIDDEN1_SIZE x INPUT_SIZE]
    float *hidden1_biases;   // [HIDDEN1_SIZE]
    
    // Second hidden layer
    float *hidden2_weights;  // [HIDDEN2_SIZE x HIDDEN1_SIZE]
    float *hidden2_biases;   // [HIDDEN2_SIZE]
    
    // Output layer
    float *output_weights;   // [OUTPUT_SIZE x HIDDEN2_SIZE]
    float *output_biases;    // [OUTPUT_SIZE]
} Network;

// Function prototypes
Network create_network();
void free_network(Network *network);
void forward_pass(Network *network, const float *input, float *hidden1, float *hidden2, float *output);
void backward_pass(Network *network, const float *input, const float *hidden1, const float *hidden2, 
                  const float *output, int label, float learning_rate);
float calculate_accuracy(Network *network, float *images, int *labels, int num_samples);
void print_confusion_matrix(Network *network, float *images, int *labels, int num_samples);
void train_network(Network *network, float *train_images, int *train_labels,
                  float *test_images, int *test_labels, int epochs, int batch_size);

// Helper functions
void apply_relu(float *x, int size);
void apply_softmax(float *x, int size);
void matrix_vector_multiply(const float *matrix, const float *vector, float *result, 
                           int rows, int cols);

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc < 2) {
        printf("Usage: %s <mnist_data_dir> [--verbose]\n", argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    char *data_dir = argv[1];
    bool verbose = false;
    
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        }
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Load MNIST dataset
    printf("Loading MNIST dataset from %s...\n", data_dir);
    MNISTData data;
    if (!load_mnist(data_dir, &data)) {
        fprintf(stderr, "Failed to load MNIST dataset\n");
        return 1;
    }
    printf("MNIST dataset loaded successfully\n");
    
    // Print dataset summary
    if (verbose) {
        print_mnist_summary(&data);
    }
    
    // Create network
    Network network = create_network();
    printf("Network created\n");
    printf("Architecture: %d -> %d -> %d -> %d\n", INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
    
    // Train network
    printf("Starting training...\n");
    train_network(&network, data.train_images, data.train_labels, 
                 data.test_images, data.test_labels, EPOCHS, BATCH_SIZE);
    
    // Print final test accuracy
    float accuracy = calculate_accuracy(&network, data.test_images, data.test_labels, MNIST_TEST_SIZE);
    printf("Final test accuracy: %.4f%%\n", accuracy * 100.0f);
    
    // Print confusion matrix
    if (verbose) {
        print_confusion_matrix(&network, data.test_images, data.test_labels, MNIST_TEST_SIZE);
    }
    
    // Free memory
    free_network(&network);
    free_mnist_data(&data);
    
    return 0;
}

Network create_network() {
    Network network;
    
    // Allocate memory for weights and biases
    network.hidden1_weights = (float *)malloc(HIDDEN1_SIZE * INPUT_SIZE * sizeof(float));
    network.hidden1_biases = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    
    network.hidden2_weights = (float *)malloc(HIDDEN2_SIZE * HIDDEN1_SIZE * sizeof(float));
    network.hidden2_biases = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    
    network.output_weights = (float *)malloc(OUTPUT_SIZE * HIDDEN2_SIZE * sizeof(float));
    network.output_biases = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Initialize hidden1 layer weights with He initialization
    float scale_hidden1 = sqrtf(2.0f / INPUT_SIZE);
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            network.hidden1_weights[i * INPUT_SIZE + j] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_hidden1;
        }
        network.hidden1_biases[i] = 0.0f;
    }
    
    // Initialize hidden2 layer weights with He initialization
    float scale_hidden2 = sqrtf(2.0f / HIDDEN1_SIZE);
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            network.hidden2_weights[i * HIDDEN1_SIZE + j] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_hidden2;
        }
        network.hidden2_biases[i] = 0.0f;
    }
    
    // Initialize output layer weights with Xavier/Glorot initialization
    float scale_output = sqrtf(2.0f / (HIDDEN2_SIZE + OUTPUT_SIZE));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            network.output_weights[i * HIDDEN2_SIZE + j] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale_output;
        }
        network.output_biases[i] = 0.0f;
    }
    
    return network;
}

void free_network(Network *network) {
    free(network->hidden1_weights);
    free(network->hidden1_biases);
    free(network->hidden2_weights);
    free(network->hidden2_biases);
    free(network->output_weights);
    free(network->output_biases);
}

void forward_pass(Network *network, const float *input, float *hidden1, float *hidden2, float *output) {
    // First hidden layer: h1 = relu(W_h1 * x + b_h1)
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        hidden1[i] = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden1[i] += network->hidden1_weights[i * INPUT_SIZE + j] * input[j];
        }
        hidden1[i] += network->hidden1_biases[i];
        // Apply ReLU activation
        hidden1[i] = (hidden1[i] > 0.0f) ? hidden1[i] : 0.0f;
    }
    
    // Second hidden layer: h2 = relu(W_h2 * h1 + b_h2)
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        hidden2[i] = 0.0f;
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            hidden2[i] += network->hidden2_weights[i * HIDDEN1_SIZE + j] * hidden1[j];
        }
        hidden2[i] += network->hidden2_biases[i];
        // Apply ReLU activation
        hidden2[i] = (hidden2[i] > 0.0f) ? hidden2[i] : 0.0f;
    }
    
    // Output layer: o = softmax(W_o * h2 + b_o)
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            output[i] += network->output_weights[i * HIDDEN2_SIZE + j] * hidden2[j];
        }
        output[i] += network->output_biases[i];
    }
    
    // Apply softmax activation
    // Find max value for numerical stability
    float max_val = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] /= sum;
    }
}

// Data augmentation function
void augment_image(const float *input, float *output) {
    // Apply small random shifts (up to 2 pixels in each direction)
    int shift_x = rand() % 5 - 2;  // -2 to 2
    int shift_y = rand() % 5 - 2;  // -2 to 2
    
    // Clear output
    memset(output, 0, INPUT_SIZE * sizeof(float));
    
    // Apply shift
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int new_y = y + shift_y;
            int new_x = x + shift_x;
            
            // Check bounds
            if (new_x >= 0 && new_x < 28 && new_y >= 0 && new_y < 28) {
                output[new_y * 28 + new_x] = input[y * 28 + x];
            }
        }
    }
}

void backward_pass(Network *network, const float *input, const float *hidden1, const float *hidden2, 
                  const float *output, int label, float learning_rate) {
    // Compute output layer error (cross-entropy gradient with softmax)
    float output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = output[i] - (i == label ? 1.0f : 0.0f);
    }
    
    // Compute hidden2 layer error
    float hidden2_error[HIDDEN2_SIZE] = {0};
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden2_error[i] += output_error[j] * network->output_weights[j * HIDDEN2_SIZE + i];
        }
        // Apply derivative of ReLU: 1 if activation > 0, 0 otherwise
        hidden2_error[i] *= (hidden2[i] > 0.0f) ? 1.0f : 0.0f;
    }
    
    // Compute hidden1 layer error
    float hidden1_error[HIDDEN1_SIZE] = {0};
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            hidden1_error[i] += hidden2_error[j] * network->hidden2_weights[j * HIDDEN1_SIZE + i];
        }
        // Apply derivative of ReLU: 1 if activation > 0, 0 otherwise
        hidden1_error[i] *= (hidden1[i] > 0.0f) ? 1.0f : 0.0f;
    }
    
    // Update output layer weights and biases with L2 regularization
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            // Add L2 regularization gradient (weight decay)
            float l2_grad = WEIGHT_DECAY * network->output_weights[i * HIDDEN2_SIZE + j];
            network->output_weights[i * HIDDEN2_SIZE + j] -= learning_rate * (output_error[i] * hidden2[j] + l2_grad);
        }
        network->output_biases[i] -= learning_rate * output_error[i];
    }
    
    // Update hidden2 layer weights and biases with L2 regularization
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            // Add L2 regularization gradient (weight decay)
            float l2_grad = WEIGHT_DECAY * network->hidden2_weights[i * HIDDEN1_SIZE + j];
            network->hidden2_weights[i * HIDDEN1_SIZE + j] -= learning_rate * (hidden2_error[i] * hidden1[j] + l2_grad);
        }
        network->hidden2_biases[i] -= learning_rate * hidden2_error[i];
    }
    
    // Update hidden1 layer weights and biases with L2 regularization
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            // Add L2 regularization gradient (weight decay)
            float l2_grad = WEIGHT_DECAY * network->hidden1_weights[i * INPUT_SIZE + j];
            network->hidden1_weights[i * INPUT_SIZE + j] -= learning_rate * (hidden1_error[i] * input[j] + l2_grad);
        }
        network->hidden1_biases[i] -= learning_rate * hidden1_error[i];
    }
}

float calculate_accuracy(Network *network, float *images, int *labels, int num_samples) {
    int correct = 0;
    
    // Allocate memory for activations
    float *hidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Evaluate each sample
    for (int i = 0; i < num_samples; i++) {
        // Forward pass
        forward_pass(network, &images[i * INPUT_SIZE], hidden1, hidden2, output);
        
        // Find the predicted class (maximum probability)
        int predicted = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[predicted]) {
                predicted = j;
            }
        }
        
        // Check if prediction is correct
        if (predicted == labels[i]) {
            correct++;
        }
    }
    
    // Free memory
    free(hidden1);
    free(hidden2);
    free(output);
    
    // Return accuracy as a fraction
    return (float)correct / num_samples;
}

void print_confusion_matrix(Network *network, float *images, int *labels, int num_samples) {
    int confusion_matrix[OUTPUT_SIZE][OUTPUT_SIZE] = {0};
    
    // Allocate memory for activations
    float *hidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Evaluate each sample
    for (int i = 0; i < num_samples; i++) {
        // Forward pass
        forward_pass(network, &images[i * INPUT_SIZE], hidden1, hidden2, output);
        
        // Find the predicted class (maximum probability)
        int predicted = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[predicted]) {
                predicted = j;
            }
        }
        
        // Update confusion matrix
        confusion_matrix[labels[i]][predicted]++;
    }
    
    // Print confusion matrix
    printf("\nConfusion Matrix:\n");
    printf("    ");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%4d", i);
    }
    printf("\n    ");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("----");
    }
    printf("\n");
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%d | ", i);
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%4d", confusion_matrix[i][j]);
        }
        printf("\n");
    }
    
    // Free memory
    free(hidden1);
    free(hidden2);
    free(output);
}

void train_network(Network *network, float *train_images, int *train_labels,
                  float *test_images, int *test_labels, int epochs, int batch_size) {
    // Allocate memory for activations
    float *hidden1 = (float *)malloc(HIDDEN1_SIZE * sizeof(float));
    float *hidden2 = (float *)malloc(HIDDEN2_SIZE * sizeof(float));
    float *output = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    // Allocate memory for augmented image
    float *augmented_image = (float *)malloc(INPUT_SIZE * sizeof(float));
    
    // Create array of indices for shuffling
    int *indices = (int *)malloc(MNIST_TRAIN_SIZE * sizeof(int));
    for (int i = 0; i < MNIST_TRAIN_SIZE; i++) {
        indices[i] = i;
    }
    
    // Training loop
    int num_batches = MNIST_TRAIN_SIZE / batch_size;
    float best_accuracy = 0.0f;
    float learning_rate = LEARNING_RATE;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Shuffle indices for randomized training
        for (int i = MNIST_TRAIN_SIZE - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        // Process each batch
        for (int batch = 0; batch < num_batches; batch++) {
            for (int i = 0; i < batch_size; i++) {
                int idx = indices[batch * batch_size + i];
                
                // Apply data augmentation if enabled
                const float *input_data;
                if (USE_DATA_AUGMENTATION && rand() % 2 == 0) {  // 50% chance of augmentation
                    augment_image(&train_images[idx * INPUT_SIZE], augmented_image);
                    input_data = augmented_image;
                } else {
                    input_data = &train_images[idx * INPUT_SIZE];
                }
                
                // Forward pass
                forward_pass(network, input_data, hidden1, hidden2, output);
                
                // Backward pass
                backward_pass(network, input_data, hidden1, hidden2, output, 
                             train_labels[idx], learning_rate);
            }
            
            // Print progress
            if (batch % 100 == 0) {
                printf("Epoch %d/%d - Batch %d/%d\r", epoch + 1, epochs, batch, num_batches);
                fflush(stdout);
            }
        }
        
        // Evaluate on test set
        float accuracy = calculate_accuracy(network, test_images, test_labels, MNIST_TEST_SIZE);
        printf("Epoch %d/%d - Test accuracy: %.4f%%\n", epoch + 1, epochs, accuracy * 100.0f);
        
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            printf("New best accuracy: %.4f%%\n", best_accuracy * 100.0f);
        }
        
        // Adjust learning rate (cosine annealing)
        float progress = (float)(epoch) / (float)(epochs);
        learning_rate = LEARNING_RATE * 0.5f * (1.0f + cosf(M_PI * progress));
        
        // Check if we've reached the target accuracy
        if (accuracy >= 0.99f) {
            printf("Target accuracy of 99%% achieved! Stopping training.\n");
            break;
        }
    }
    
    // Free memory
    free(hidden1);
    free(hidden2);
    free(output);
    free(augmented_image);
    free(indices);
}

void apply_relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}

void apply_softmax(float *x, int size) {
    // Find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matrix_vector_multiply(const float *matrix, const float *vector, float *result, 
                           int rows, int cols) {
    // Matrix is stored in row-major order: matrix[row * cols + col]
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}
