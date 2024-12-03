#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "merge.h"
#include "linalg.h"

// Create merge accumulator
MergeAccumulator* create_merge_accumulator(int height, int width, int channels) {
    MergeAccumulator* acc = (MergeAccumulator*)malloc(sizeof(MergeAccumulator));
    if (!acc) return NULL;

    acc->height = height;
    acc->width = width;
    acc->channels = channels;
    size_t size = height * width * channels * sizeof(float);

    acc->numerator = (float*)calloc(height * width * channels, sizeof(float));
    acc->denominator = (float*)calloc(height * width * channels, sizeof(float));

    if (!acc->numerator || !acc->denominator) {
        free_merge_accumulator(acc);
        return NULL;
    }

    return acc;
}

// Free merge accumulator
void free_merge_accumulator(MergeAccumulator* acc) {
    if (acc) {
        free(acc->numerator);
        free(acc->denominator);
        free(acc);
    }
}

// Compute merge weights based on kernels and robustness
void compute_merge_weights(const Image* image,
                         const SteerableKernels* kernels,
                         const float* robustness,
                         float* weights,
                         const MergeParams* params) {
    const int height = image->height;
    const int width = image->width;
    const int kernel_size = kernels->size;
    const int radius = kernel_size / 2;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float max_response = -FLT_MAX;
            int best_kernel = 0;

            // Find best matching kernel orientation
            if (params->use_kernels) {
                for (int k = 0; k < kernels->count; k++) {
                    float response = 0.0f;
                    const float* kernel = kernels->weights + k * kernel_size * kernel_size;

                    // Compute kernel response
                    for (int dy = -radius; dy <= radius; dy++) {
                        for (int dx = -radius; dx <= radius; dx++) {
                            int ny = y + dy;
                            int nx = x + dx;
                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                float kernel_val = kernel[(dy + radius) * kernel_size + (dx + radius)];
                                float image_val = 0.0f;
                                for (int c = 0; c < image->channels; c++) {
                                    image_val += image->data[(ny * width + nx) * image->channels + c];
                                }
                                response += kernel_val * image_val;
                            }
                        }
                    }

                    if (response > max_response) {
                        max_response = response;
                        best_kernel = k;
                    }
                }
            }

            // Compute final weight
            float weight = 1.0f;
            if (params->use_kernels) {
                weight *= max_response;
            }
            if (params->use_robustness && robustness) {
                weight *= robustness[y * width + x];
            }

            // Apply power and noise model
            weight = powf(weight, params->power_max);
            if (params->noise_sigma > 0.0f) {
                float noise_factor = 1.0f / (1.0f + params->noise_sigma * params->noise_sigma);
                weight *= noise_factor;
            }

            weights[y * width + x] = weight;
        }
    }
}

// Merge image into accumulator
void merge_image(const Image* image,
                const SteerableKernels* kernels,
                const float* robustness,
                MergeAccumulator* acc,
                const MergeParams* params) {
    if (!image || !acc || !params) return;

    const int height = image->height;
    const int width = image->width;
    const int channels = image->channels;

    // Compute weights
    float* weights = (float*)malloc(height * width * sizeof(float));
    if (!weights) return;

    compute_merge_weights(image, kernels, robustness, weights, params);

    // Accumulate weighted image
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels + c;
                float weight = weights[y * width + x];
                acc->numerator[idx] += weight * image->data[idx];
                acc->denominator[idx] += weight;
            }
        }
    }

    free(weights);
}

// Merge reference image
void merge_reference(const Image* ref_image,
                    MergeAccumulator* acc,
                    const MergeParams* params) {
    if (!ref_image || !acc || !params) return;

    // For reference image, use simplified merging with uniform weights
    const int height = ref_image->height;
    const int width = ref_image->width;
    const int channels = ref_image->channels;

    float ref_weight = params->power_max;  // Use maximum weight for reference
    if (params->noise_sigma > 0.0f) {
        ref_weight *= 1.0f / (1.0f + params->noise_sigma * params->noise_sigma);
    }

    #pragma omp parallel for collapse(3)
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * channels + c;
                acc->numerator[idx] += ref_weight * ref_image->data[idx];
                acc->denominator[idx] += ref_weight;
            }
        }
    }
}

// Normalize accumulator and create final image
void normalize_accumulator(MergeAccumulator* acc) {
    if (!acc) return;

    const float epsilon = 1e-10f;
    const size_t total_size = acc->height * acc->width * acc->channels;

    #pragma omp parallel for
    for (size_t i = 0; i < total_size; i++) {
        float denom = acc->denominator[i];
        if (denom > epsilon) {
            acc->numerator[i] /= denom;
        }
    }
}

// Reconstruct final merged image
Image* reconstruct_merged_image(const MergeAccumulator* acc) {
    if (!acc) return NULL;

    Image* result = (Image*)malloc(sizeof(Image));
    if (!result) return NULL;

    result->height = acc->height;
    result->width = acc->width;
    result->channels = acc->channels;
    size_t size = acc->height * acc->width * acc->channels;

    result->data = (float*)malloc(size * sizeof(float));
    if (!result->data) {
        free(result);
        return NULL;
    }

    // Copy normalized accumulator to result
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        result->data[i] = acc->numerator[i];
    }

    return result;
} 