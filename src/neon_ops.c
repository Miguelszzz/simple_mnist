/**
 * @file neon_ops.c
 * @brief Implementation of ARM Neon optimized operations
 */

#include "neon_ops.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Check if we're on an ARM platform and include ARM Neon headers
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

bool neon_available(void) {
#if HAVE_NEON
    return true;
#else
    return false;
#endif
}

// Fallback implementation for non-ARM platforms
static void __attribute__((unused)) fallback_matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K) {
    // Initialize C to zeros
    memset(C, 0, M * N * sizeof(float));
    
    // Compute C = A * B
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void neon_matrix_multiply(const float *A, const float *B, float *C, int M, int N, int K) {
#if HAVE_NEON
    // Initialize C to zeros
    memset(C, 0, M * N * sizeof(float));
    
    // Process 4 rows of A and 4 columns of B at a time when possible
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                // 4 output values at a time
                float32x4_t c_val = vdupq_n_f32(0.0f);
                
                for (int k = 0; k < K; k++) {
                    float32x4_t b_val = vld1q_f32(&B[k * N + j]);
                    float32x4_t a_val = vdupq_n_f32(A[i * K + k]);
                    c_val = vmlaq_f32(c_val, a_val, b_val);
                }
                
                vst1q_f32(&C[i * N + j], c_val);
            } else {
                // Handle remaining columns (less than 4)
                for (int jj = j; jj < N; jj++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + jj];
                    }
                    C[i * N + jj] = sum;
                }
            }
        }
    }
#else
    fallback_matrix_multiply(A, B, C, M, N, K);
#endif
}

void neon_matrix_vector_multiply(const float *A, const float *x, float *y, int M, int N) {
#if HAVE_NEON
    // Process 4 elements at a time
    for (int i = 0; i < M; i++) {
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        int j = 0;
        
        // Process 4 elements at a time
        for (; j <= N - 4; j += 4) {
            float32x4_t a_vec = vld1q_f32(&A[i * N + j]);
            float32x4_t x_vec = vld1q_f32(&x[j]);
            sum_vec = vmlaq_f32(sum_vec, a_vec, x_vec);
        }
        
        // Extract the sum
        float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                    vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
        
        // Handle remaining elements
        for (; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        
        y[i] = sum;
    }
#else
    // Fallback implementation
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
#endif
}

void neon_relu(const float *x, float *y, int n) {
#if HAVE_NEON
    const float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    
    // Process 4 elements at a time
    for (; i <= n - 4; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x[i]);
        float32x4_t y_vec = vmaxq_f32(x_vec, zero);
        vst1q_f32(&y[i], y_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        y[i] = fmaxf(x[i], 0.0f);
    }
#else
    // Fallback implementation
    for (int i = 0; i < n; i++) {
        y[i] = fmaxf(x[i], 0.0f);
    }
#endif
}

void neon_elementwise_multiply(const float *x, const float *y, float *z, int n) {
#if HAVE_NEON
    int i = 0;
    
    // Process 4 elements at a time
    for (; i <= n - 4; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x[i]);
        float32x4_t y_vec = vld1q_f32(&y[i]);
        float32x4_t z_vec = vmulq_f32(x_vec, y_vec);
        vst1q_f32(&z[i], z_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        z[i] = x[i] * y[i];
    }
#else
    // Fallback implementation
    for (int i = 0; i < n; i++) {
        z[i] = x[i] * y[i];
    }
#endif
}

// Fast approximation of exp function using ARM Neon
static float32x4_t exp_ps(float32x4_t x) {
#if HAVE_NEON
    // Better exp approximation using minimax polynomial
    const float32x4_t LOG2EF = vdupq_n_f32(1.442695040f);
    const float32x4_t C1 = vdupq_n_f32(0.693147182f);
    const float32x4_t C2 = vdupq_n_f32(0.240226337f);
    const float32x4_t C3 = vdupq_n_f32(0.055504110f);
    const float32x4_t C4 = vdupq_n_f32(0.009618129f);
    const float32x4_t C5 = vdupq_n_f32(0.001333355f);

    // Clamp input to avoid overflow
    const float32x4_t max_val = vdupq_n_f32(88.3762626647949f);
    const float32x4_t min_val = vdupq_n_f32(-88.3762626647949f);
    x = vminq_f32(vmaxq_f32(x, min_val), max_val);

    // Scale by log2(e)
    float32x4_t z = vmulq_f32(x, LOG2EF);

    // Round to nearest integer
    float32x4_t n = vcvtq_f32_s32(vcvtq_s32_f32(z));

    // Polynomial approximation of exp2(fractional part)
    float32x4_t p = vsubq_f32(z, n);
    float32x4_t xx = vmulq_f32(p, p);
    float32x4_t px = p;
    float32x4_t result = vaddq_f32(C1, vmulq_f32(C2, px));
    px = vmulq_f32(px, xx);
    result = vaddq_f32(result, vmulq_f32(C3, px));
    px = vmulq_f32(px, xx);
    result = vaddq_f32(result, vmulq_f32(C4, px));
    px = vmulq_f32(px, xx);
    result = vaddq_f32(result, vmulq_f32(C5, px));

    // Reconstruct exp(x) = 2^n * exp2(fractional part)
    const int32x4_t pow2n = vshlq_n_s32(vcvtq_s32_f32(n), 23);
    const float32x4_t pow2 = vreinterpretq_f32_s32(vaddq_s32(pow2n, vdupq_n_s32(0x3f800000)));
    result = vmulq_f32(result, pow2);
    
    return result;
#else
    // This should never be called in non-NEON code paths
    return vdupq_n_f32(0.0f);
#endif
}

void neon_exp(const float *x, float *y, int n) {
#if HAVE_NEON
    int i = 0;
    
    // Process 4 elements at a time
    for (; i <= n - 4; i += 4) {
        float32x4_t x_vec = vld1q_f32(&x[i]);
        float32x4_t y_vec = exp_ps(x_vec);
        vst1q_f32(&y[i], y_vec);
    }
    
    // Handle remaining elements with standard exp
    for (; i < n; i++) {
        y[i] = expf(x[i]);
    }
#else
    // Fallback implementation
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i]);
    }
#endif
}

void neon_softmax(const float *x, float *y, int n) {
#if HAVE_NEON
    // Find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    float *exp_values = (float *)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        exp_values[i] = x[i] - max_val;
    }
    
    neon_exp(exp_values, exp_values, n);
    
    for (int i = 0; i < n; i++) {
        sum += exp_values[i];
    }
    
    // Normalize to get probabilities
    float32x4_t sum_vec = vdupq_n_f32(sum);
    int i = 0;
    
    // Process 4 elements at a time
    for (; i <= n - 4; i += 4) {
        float32x4_t exp_vec = vld1q_f32(&exp_values[i]);
        float32x4_t y_vec = vdivq_f32(exp_vec, sum_vec);
        vst1q_f32(&y[i], y_vec);
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        y[i] = exp_values[i] / sum;
    }
    
    free(exp_values);
#else
    // Fallback implementation
    // Find max value for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }
    
    // Normalize to get probabilities
    for (int i = 0; i < n; i++) {
        y[i] /= sum;
    }
#endif
}
