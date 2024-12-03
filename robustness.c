#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "robustness.h"
#include "linalg.h"

// Helper function to compute Dogson quadratic kernel
float dogson_quadratic_kernel(float x) {
    float abs_x = fabsf(x);
    if (abs_x <= 0.5f) {
        return -2.0f * abs_x * abs_x + 1.0f;
    } else if (abs_x <= 1.5f) {
        return abs_x * abs_x - 2.5f * abs_x + 1.5f;
    }
    return 0.0f;
}

// Helper function to compute Dogson biquadratic kernel
float dogson_biquadratic_kernel(float x, float y) {
    return dogson_quadratic_kernel(x) * dogson_quadratic_kernel(y);
}

// Initialize robustness computation
LocalStats* init_robustness(const Image* image, const RobustnessParams* params) {
    if (!image || !params || !params->enabled) {
        return NULL;
    }

    // Allocate local statistics structure
    LocalStats* stats = (LocalStats*)malloc(sizeof(LocalStats));
    if (!stats) {
        return NULL;
    }

    stats->height = image->height;
    stats->width = image->width;
    stats->channels = image->channels;

    // Allocate memory for means and standard deviations
    size_t size = stats->height * stats->width * stats->channels * sizeof(float);
    stats->means = (float*)malloc(size);
    stats->stds = (float*)malloc(size);

    if (!stats->means || !stats->stds) {
        free_local_stats(stats);
        return NULL;
    }

    // Compute local statistics
    compute_local_statistics(image, params, stats->means, stats->stds);
    return stats;
}

// Free local statistics
void free_local_stats(LocalStats* stats) {
    if (stats) {
        free(stats->means);
        free(stats->stds);
        free(stats);
    }
}

// Compute local statistics for each pixel
void compute_local_statistics(const Image* image, 
                            const RobustnessParams* params,
                            float* means, 
                            float* stds) {
    const int window_size = params->window_size;
    const int radius = window_size / 2;
    
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < image->channels; c++) {
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                float sum = 0.0f;
                float sum_sq = 0.0f;
                int count = 0;

                // Compute local mean and variance in window
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;

                        if (ny >= 0 && ny < image->height && 
                            nx >= 0 && nx < image->width) {
                            float val = image->data[(ny * image->width + nx) * image->channels + c];
                            sum += val;
                            sum_sq += val * val;
                            count++;
                        }
                    }
                }

                // Store mean and variance
                int idx = (y * image->width + x) * image->channels + c;
                means[idx] = sum / count;
                stds[idx] = (sum_sq / count) - (means[idx] * means[idx]);
            }
        }
    }
}

// Compute robustness map
float* compute_robustness(const Image* image,
                         const LocalStats* ref_stats,
                         const AlignmentMap* alignment,
                         const RobustnessParams* params) {
    if (!image || !ref_stats || !alignment || !params || !params->enabled) {
        return NULL;
    }

    // Allocate memory for intermediate results
    const int height = image->height;
    const int width = image->width;
    float* d_sq = (float*)malloc(height * width * sizeof(float));
    float* sigma_sq = (float*)malloc(height * width * sizeof(float));
    float* S = (float*)malloc(height * width * sizeof(float));
    float* R = (float*)malloc(height * width * sizeof(float));
    float* r = (float*)malloc(height * width * sizeof(float));

    if (!d_sq || !sigma_sq || !S || !R || !r) {
        free(d_sq);
        free(sigma_sq);
        free(S);
        free(R);
        free(r);
        return NULL;
    }

    // Compute color distances and noise model
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float d_sum = 0.0f;
            float sigma_sum = 0.0f;

            // Get aligned position
            float aligned_y = y + alignment->data[y * width + x].y;
            float aligned_x = x + alignment->data[y * width + x].x;

            // Compute color distance and noise model for each channel
            for (int c = 0; c < image->channels; c++) {
                float ref_mean = bilinear_interpolate(ref_stats->means + c * height * width,
                                                    height, width, aligned_y, aligned_x);
                float ref_std = bilinear_interpolate(ref_stats->stds + c * height * width,
                                                   height, width, aligned_y, aligned_x);

                float img_val = image->data[(y * width + x) * image->channels + c];
                float d = fabsf(img_val - ref_mean);
                float d_norm = d * d / (ref_std + params->epsilon);

                d_sum += d_norm;
                sigma_sum += ref_std;
            }

            d_sq[y * width + x] = d_sum;
            sigma_sq[y * width + x] = sigma_sum;
        }
    }

    // Compute flow irregularity penalty
    compute_flow_irregularity(alignment, params, S);

    // Apply robustness threshold
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float s = S[y * width + x];
            float d = d_sq[y * width + x];
            float sigma = sigma_sq[y * width + x];
            
            R[y * width + x] = fmaxf(0.0f, 
                                    fminf(1.0f, 
                                         s * expf(-d / (sigma + params->epsilon)) - params->threshold));
        }
    }

    // Compute local minimum in 5x5 window
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float min_val = FLT_MAX;

            for (int dy = -2; dy <= 2; dy++) {
                for (int dx = -2; dx <= 2; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        min_val = fminf(min_val, R[ny * width + nx]);
                    }
                }
            }

            r[y * width + x] = min_val;
        }
    }

    // Cleanup
    free(d_sq);
    free(sigma_sq);
    free(S);
    free(R);

    return r;
}

// Helper function to compute flow irregularity
static void compute_flow_irregularity(const AlignmentMap* flow,
                                    const RobustnessParams* params,
                                    float* S) {
    const int height = flow->height;
    const int width = flow->width;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float min_x = FLT_MAX, max_x = -FLT_MAX;
            float min_y = FLT_MAX, max_y = -FLT_MAX;

            // Find min/max flow in 3x3 neighborhood
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        const Alignment* flow_val = &flow->data[ny * width + nx];
                        min_x = fminf(min_x, flow_val->x);
                        max_x = fmaxf(max_x, flow_val->x);
                        min_y = fminf(min_y, flow_val->y);
                        max_y = fmaxf(max_y, flow_val->y);
                    }
                }
            }

            float diff_x = max_x - min_x;
            float diff_y = max_y - min_y;
            float diff_sq = diff_x * diff_x + diff_y * diff_y;

            // Apply flow irregularity threshold
            S[y * width + x] = (diff_sq > params->mt * params->mt) ? params->s1 : params->s2;
        }
    }
} 