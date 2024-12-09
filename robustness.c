#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "robustness.h"
#include "linalg.h"

// Add debug logging and error checking macros
#ifdef DEBUG_ROBUSTNESS
#define LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

#define CHECK_NULL(ptr, msg) do { \
    if (!(ptr)) { \
        fprintf(stderr, "Error: %s is NULL\n", msg); \
        return NULL; \
    } \
} while(0)

#define CHECK_ALLOC(ptr, msg) do { \
    if (!(ptr)) { \
        fprintf(stderr, "Error: Failed to allocate memory for %s\n", msg); \
        return NULL; \
    } \
} while(0)

// Add validation helper
static bool validate_params(const RobustnessParams* params) {
    if (!params) {
        fprintf(stderr, "Error: NULL robustness parameters\n");
        return false;
    }
    
    if (params->window_size <= 0 || params->window_size % 2 == 0) {
        fprintf(stderr, "Error: Invalid window size %d (must be positive odd)\n", 
                params->window_size);
        return false;
    }
    
    if (params->epsilon <= 0.0f) {
        fprintf(stderr, "Error: Invalid epsilon %f (must be positive)\n", 
                params->epsilon);
        return false;
    }
    
    if (params->bayer_mode && !params->cfa_pattern) {
        fprintf(stderr, "Error: Bayer mode enabled but no CFA pattern provided\n");
        return false;
    }
    
    if (params->noise.curve_size <= 0) {
        fprintf(stderr, "Error: Invalid noise curve size %d\n", 
                params->noise.curve_size);
        return false;
    }
    
    return true;
}

// Add image validation helper
static bool validate_image(const Image* img, const char* name) {
    if (!img) {
        fprintf(stderr, "Error: NULL %s image\n", name);
        return false;
    }
    
    if (!img->data) {
        fprintf(stderr, "Error: NULL data in %s image\n", name);
        return false;
    }
    
    if (img->width <= 0 || img->height <= 0) {
        fprintf(stderr, "Error: Invalid dimensions %dx%d in %s image\n",
                img->width, img->height, name);
        return false;
    }
    
    return true;
}

// Add noise model validation
static bool validate_noise_model(const NoiseModel* noise) {
    if (!noise) {
        fprintf(stderr, "Error: NULL noise model\n");
        return false;
    }
    
    if (!noise->std_curve || !noise->diff_curve) {
        fprintf(stderr, "Error: NULL noise curves\n");
        return false;
    }
    
    if (noise->curve_size <= 0) {
        fprintf(stderr, "Error: Invalid noise curve size %d\n", noise->curve_size);
        return false;
    }
    
    return true;
}

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
    // Validate inputs
    if (!validate_image(image, "input") || !validate_params(params)) {
        return NULL;
    }
    
    if (!params->enabled) {
        LOG_DEBUG("Robustness disabled, returning NULL");
        return NULL;
    }

    // Allocate local statistics structure
    LocalStats* stats = (LocalStats*)malloc(sizeof(LocalStats));
    CHECK_ALLOC(stats, "local statistics");

    stats->height = image->height;
    stats->width = image->width;
    stats->channels = image->channels;

    // Allocate memory for means and standard deviations
    size_t size = stats->height * stats->width * stats->channels * sizeof(float);
    stats->means = (float*)malloc(size);
    stats->stds = (float*)malloc(size);

    if (!stats->means || !stats->stds) {
        fprintf(stderr, "Error: Failed to allocate memory for statistics arrays\n");
        free_local_stats(stats);
        return NULL;
    }

    // Compute local statistics
    compute_local_statistics(image, params, stats->means, stats->stds);
    
    LOG_DEBUG("Initialized robustness for %dx%d image with %d channels",
              stats->width, stats->height, stats->channels);
    
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
float* compute_robustness(const Image* image, const LocalStats* ref_stats,
                         const AlignmentMap* alignment, const RobustnessParams* params) {
    // Validate all inputs
    if (!validate_image(image, "input") || 
        !validate_params(params) ||
        !alignment) {
        return NULL;
    }
    
    if (!params->enabled) {
        LOG_DEBUG("Robustness disabled, returning NULL");
        return NULL;
    }
    
    if (!ref_stats || !ref_stats->means || !ref_stats->stds) {
        fprintf(stderr, "Error: Invalid reference statistics\n");
        return NULL;
    }
    
    // Validate dimensions match
    if (image->height != ref_stats->height || 
        image->width != ref_stats->width ||
        image->channels != ref_stats->channels) {
        fprintf(stderr, "Error: Dimension mismatch between image and reference stats\n");
        return NULL;
    }

    // Allocate memory for intermediate results with error checking
    const int height = image->height;
    const int width = image->width;
    float* d_sq = (float*)malloc(height * width * sizeof(float));
    float* sigma_sq = (float*)malloc(height * width * sizeof(float));
    float* S = (float*)malloc(height * width * sizeof(float));
    float* R = (float*)malloc(height * width * sizeof(float));
    float* r = (float*)malloc(height * width * sizeof(float));

    if (!d_sq || !sigma_sq || !S || !R || !r) {
        fprintf(stderr, "Error: Failed to allocate memory for robustness computation\n");
        free(d_sq); free(sigma_sq); free(S); free(R); free(r);
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

    LOG_DEBUG("Computing robustness for %dx%d image", width, height);
    
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

GuideImage* compute_guide_image(const Image* raw_img, const int* cfa_pattern) {
    if (!raw_img || !cfa_pattern) return NULL;

    GuideImage* guide = (GuideImage*)malloc(sizeof(GuideImage));
    if (!guide) return NULL;

    // For Bayer pattern, output is half size with 3 channels
    guide->height = raw_img->height / 2;
    guide->width = raw_img->width / 2;
    guide->channels = 3;
    guide->data = (float*)malloc(guide->height * guide->width * guide->channels * sizeof(float));

    if (!guide->data) {
        free(guide);
        return NULL;
    }

    // Process each 2x2 Bayer quad
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < guide->height; y++) {
        for (int x = 0; x < guide->width; x++) {
            float g = 0.0f;
            
            // Process each pixel in the quad
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    int raw_y = 2*y + i;
                    int raw_x = 2*x + j;
                    float val = raw_img->data[raw_y * raw_img->width + raw_x];
                    
                    int color = cfa_pattern[i * 2 + j];
                    if (color == 1) { // Green
                        g += val;
                    } else { // Red or Blue
                        guide->data[(y * guide->width + x) * 3 + color] = val;
                    }
                }
            }
            
            // Store average green
            guide->data[(y * guide->width + x) * 3 + 1] = g / 2.0f;
        }
    }

    return guide;
}

void free_guide_image(GuideImage* guide) {
    if (guide) {
        free(guide->data);
        free(guide);
    }
}

void apply_noise_model(const float* d_p, const float* ref_means, const float* ref_stds,
                      const NoiseModel* noise, float* d_sq, float* sigma_sq,
                      int height, int width, int channels) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float d_sq_sum = 0.0f;
            float sigma_sq_sum = 0.0f;

            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                float brightness = ref_means[idx];
                int noise_idx = (int)(1000.0f * brightness);
                if (noise_idx >= noise->curve_size) noise_idx = noise->curve_size - 1;

                float d_t = noise->diff_curve[noise_idx];
                float sigma_t = noise->std_curve[noise_idx];
                float sigma_p_sq = ref_stds[idx];
                
                sigma_sq_sum += fmaxf(sigma_p_sq, sigma_t * sigma_t);

                float d_p_val = d_p[idx];
                float d_p_sq = d_p_val * d_p_val;
                float shrink = d_p_sq / (d_p_sq + d_t * d_t);
                d_sq_sum += d_p_sq * shrink * shrink;
            }

            d_sq[y * width + x] = d_sq_sum;
            sigma_sq[y * width + x] = sigma_sq_sum;
        }
    }
} 