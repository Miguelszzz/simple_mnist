/**
 * @file neon_ops.h
 * @brief ARM Neon optimized operations for neural network computations
 * 
 * This file contains optimized implementations of common neural network
 * operations using ARM Neon SIMD instructions, specifically targeting
 * Apple Silicon (M1/M2/M3) processors.
 */

#ifndef NEON_OPS_H
#define NEON_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/**
 * @brief Detect if ARM Neon is available at runtime
 * 
 * @return true if ARM Neon is available, false otherwise
 */
bool neon_available(void);

/**
 * @brief Matrix multiplication optimized with ARM Neon
 * 
 * Computes C = A * B where:
 * A is an M x K matrix
 * B is a K x N matrix
 * C is an M x N matrix
 * 
 * @param A Pointer to matrix A (row-major)
 * @param B Pointer to matrix B (row-major)
 * @param C Pointer to output matrix C (row-major)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
void neon_matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K);

/**
 * @brief Matrix-vector multiplication optimized with ARM Neon
 * 
 * Computes y = A * x where:
 * A is an M x N matrix
 * x is a vector of length N
 * y is a vector of length M
 * 
 * @param A Pointer to matrix A (row-major)
 * @param x Pointer to vector x
 * @param y Pointer to output vector y
 * @param M Number of rows in A
 * @param N Number of columns in A
 */
void neon_matrix_vector_multiply(const float *A, const float *x, float *y, int M, int N);

/**
 * @brief Apply ReLU activation function using ARM Neon
 * 
 * Computes ReLU(x) element-wise on input vector
 * 
 * @param x Pointer to input vector
 * @param y Pointer to output vector (can be the same as x for in-place)
 * @param n Length of vectors
 */
void neon_relu(const float *x, float *y, int n);

/**
 * @brief Apply elementwise multiply using ARM Neon
 * 
 * Computes z[i] = x[i] * y[i] for all i
 * 
 * @param x Pointer to first input vector
 * @param y Pointer to second input vector
 * @param z Pointer to output vector
 * @param n Length of vectors
 */
void neon_elementwise_multiply(const float *x, const float *y, float *z, int n);

/**
 * @brief Compute exponential function using ARM Neon
 * 
 * Computes y[i] = exp(x[i]) for all i
 * 
 * @param x Pointer to input vector
 * @param y Pointer to output vector
 * @param n Length of vectors
 */
void neon_exp(const float *x, float *y, int n);

/**
 * @brief Compute softmax function using ARM Neon
 * 
 * Computes softmax of input vector x
 * 
 * @param x Pointer to input vector
 * @param y Pointer to output vector
 * @param n Length of vectors
 */
void neon_softmax(const float *x, float *y, int n);


#endif /* NEON_OPS_H */
