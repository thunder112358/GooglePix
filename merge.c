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
void merge_image(const Image* image, const SteerableKernels* kernels,
                const float* robustness, MergeAccumulator* acc,
                const MergeParams* params) {
    if (!image || !acc || !params) return;

    const int height = image->height;
    const int width = image->width;
    const int channels = image->channels;
    const float scale = params->scale;
    const int output_height = (int)(height * scale);
    const int output_width = (int)(width * scale);

    // Process each output pixel
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            // Compute input image position
            float ref_y = y / scale;
            float ref_x = x / scale;
            
            // Get patch position
            int patch_y = (int)(ref_y / params->tile_size);
            int patch_x = (int)(ref_x / params->tile_size);

            // Get robustness value
            float rob = robustness ? robustness[(int)(ref_y + 0.5f) * width + (int)(ref_x + 0.5f)] : 1.0f;

            // For each channel
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                float weight_sum = 0.0f;

                // Process neighborhood
                int center_x = (int)(ref_x + 0.5f);
                int center_y = (int)(ref_y + 0.5f);
                int rad = 1;

                for (int dy = -rad; dy <= rad; dy++) {
                    for (int dx = -rad; dx <= rad; dx++) {
                        int px = center_x + dx;
                        int py = center_y + dy;

                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            float val = image->data[(py * width + px) * channels + c];
                            
                            // Compute distance and weight
                            float dist_x = px - ref_x;
                            float dist_y = py - ref_y;
                            float dist_sq = dist_x * dist_x + dist_y * dist_y;
                            
                            float weight;
                            if (params->iso_kernel) {
                                weight = expf(-2.0f * dist_sq);
                            } else {
                                // Use anisotropic kernel if available
                                weight = expf(-0.5f * dist_sq);
                            }

                            weight *= rob;  // Apply robustness

                            sum += val * weight;
                            weight_sum += weight;
                        }
                    }
                }

                // Accumulate
                int out_idx = (y * output_width + x) * channels + c;
                acc->numerator[out_idx] += sum;
                acc->denominator[out_idx] += weight_sum;
            }
        }
    }
}

// Helper functions for merge
static float denoise_power_merge(float acc_rob, float max_multiplier, int max_frame_count) {
    if (acc_rob >= max_frame_count) return 1.0f;
    return max_multiplier - (max_multiplier - 1.0f) * acc_rob / max_frame_count;
}

static float denoise_range_merge(float acc_rob, float rad_max, int max_frame_count) {
    if (acc_rob >= max_frame_count) return 1.0f;
    return rad_max - (rad_max - 1.0f) * acc_rob / max_frame_count;
}

// Merge reference image
void merge_reference(const Image* ref_img, MergeAccumulator* acc, const MergeParams* params) {
    if (!ref_img || !acc || !params) return;

    const int height = ref_img->height;
    const int width = ref_img->width;
    const int channels = ref_img->channels;
    const float scale = params->scale;
    const int output_height = (int)(height * scale);
    const int output_width = (int)(width * scale);

    // Process each output pixel
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < output_height; y++) {
        for (int x = 0; x < output_width; x++) {
            // Compute input image position
            float ref_y = y / scale;
            float ref_x = x / scale;
            
            // For each channel
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;
                float weight_sum = 0.0f;

                // Process neighborhood
                int center_x = (int)(ref_x + 0.5f);
                int center_y = (int)(ref_y + 0.5f);
                int rad = 1;  // Default radius for reference image

                for (int dy = -rad; dy <= rad; dy++) {
                    for (int dx = -rad; dx <= rad; dx++) {
                        int px = center_x + dx;
                        int py = center_y + dy;

                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            float val = ref_img->data[(py * width + px) * channels + c];
                            
                            // Compute distance and weight
                            float dist_x = px - ref_x;
                            float dist_y = py - ref_y;
                            float dist_sq = dist_x * dist_x + dist_y * dist_y;
                            
                            float weight;
                            if (params->iso_kernel) {
                                weight = expf(-2.0f * dist_sq);
                            } else {
                                // Use anisotropic kernel if available
                                weight = expf(-0.5f * dist_sq);
                            }

                            sum += val * weight;
                            weight_sum += weight;
                        }
                    }
                }

                // Accumulate
                int out_idx = (y * output_width + x) * channels + c;
                acc->numerator[out_idx] += sum;
                acc->denominator[out_idx] += weight_sum;
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