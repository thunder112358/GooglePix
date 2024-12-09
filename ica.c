#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "ica.h"
#include "block_matching.h"

// Add function declaration at the top
static float clamp(float x, float min_val, float max_val);

// Helper function declarations
static void compute_prewitt_gradients(const Image* img, ImageGradients* grads);
static void apply_gaussian_blur(Image* img, float sigma);
static void compute_gaussian_kernel_1d(float* kernel, int size, float sigma);
static float bilinear_interpolate(const Image* img, float x, float y);
// Helper functions for debug output
static float find_min(const float* data, int size) {
    float min_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
    }
    return min_val;
}

static float find_max(const float* data, int size) {
    float max_val = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    return max_val;
} 


// Memory management functions
ImageGradients* create_image_gradients(int height, int width) {
    ImageGradients* grads = (ImageGradients*)malloc(sizeof(ImageGradients));
    if (!grads) return NULL;

    grads->height = height;
    grads->width = width;
    grads->data_x = (pixel_t*)malloc(sizeof(pixel_t) * height * width);
    grads->data_y = (pixel_t*)malloc(sizeof(pixel_t) * height * width);

    if (!grads->data_x || !grads->data_y) {
        free_image_gradients(grads);
        return NULL;
    }

    return grads;
}

void free_image_gradients(ImageGradients* grads) {
    if (grads) {
        free(grads->data_x);
        free(grads->data_y);
        free(grads);
    }
}

HessianMatrix* create_hessian_matrix(int height, int width) {
    HessianMatrix* hessian = (HessianMatrix*)malloc(sizeof(HessianMatrix));
    if (!hessian) return NULL;

    hessian->height = height;
    hessian->width = width;
    // 4 elements per 2x2 matrix
    hessian->data = (float*)calloc(height * width * 4, sizeof(float));

    if (!hessian->data) {
        free(hessian);
        return NULL;
    }

    return hessian;
}

void free_hessian_matrix(HessianMatrix* hessian) {
    if (hessian) {
        free(hessian->data);
        free(hessian);
    }
}

// Gradient computation
static void compute_prewitt_gradients(const Image* img, ImageGradients* grads) {
    // Prewitt kernels
    const float kernelx[3] = {-1.0f, 0.0f, 1.0f};
    const float kernely[3] = {-1.0f, 0.0f, 1.0f};

    // Compute gradients with circular padding
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum_x = 0.0f;
            float sum_y = 0.0f;

            // Apply horizontal kernel
            for (int k = -1; k <= 1; k++) {
                int xk = x + k;
                // Circular padding
                if (xk < 0) xk += img->width;
                if (xk >= img->width) xk -= img->width;
                sum_x += img->data[y * img->width + xk] * kernelx[k + 1];
            }

            // Apply vertical kernel
            for (int k = -1; k <= 1; k++) {
                int yk = y + k;
                // Circular padding
                if (yk < 0) yk += img->height;
                if (yk >= img->height) yk -= img->height;
                sum_y += img->data[yk * img->width + x] * kernely[k + 1];
            }

            grads->data_x[y * img->width + x] = sum_x;
            grads->data_y[y * img->width + x] = sum_y;
        }
    }
}

// Gaussian blur implementation
static void compute_gaussian_kernel_1d(float* kernel, int size, float sigma) {
    int center = size / 2;
    float sum = 0.0f;

    // Match Python's _gaussian_kernel1d
    for (int i = 0; i < size; i++) {
        float x = (float)(i - center);
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

static void apply_gaussian_blur(Image* img, float sigma) {
    if (sigma <= 0.0f) return;

    // Match Python's kernel radius calculation
    int kernel_radius = (int)(4.0f * sigma + 0.5f);
    int kernel_size = 2 * kernel_radius + 1;
    
    // Allocate kernel and temporary buffer
    float* kernel = (float*)malloc(sizeof(float) * kernel_size);
    float* temp = (float*)malloc(sizeof(float) * img->height * img->width);
    
    if (!kernel || !temp) {
        free(kernel);
        free(temp);
        return;
    }

    compute_gaussian_kernel_1d(kernel, kernel_size, sigma);

    // Horizontal pass (separable convolution)
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -kernel_radius; k <= kernel_radius; k++) {
                int xk = x + k;
                // Use circular padding to match PyTorch's 'circular' mode
                if (xk < 0) xk += img->width;
                if (xk >= img->width) xk -= img->width;
                
                float weight = kernel[k + kernel_radius];
                sum += img->data[y * img->width + xk] * weight;
                weight_sum += weight;
            }

            temp[y * img->width + x] = sum / weight_sum;
        }
    }

    // Vertical pass
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -kernel_radius; k <= kernel_radius; k++) {
                int yk = y + k;
                // Use circular padding
                if (yk < 0) yk += img->height;
                if (yk >= img->height) yk -= img->height;
                
                float weight = kernel[k + kernel_radius];
                sum += temp[yk * img->width + x] * weight;
                weight_sum += weight;
            }

            img->data[y * img->width + x] = sum / weight_sum;
        }
    }

    free(kernel);
    free(temp);
}

// Bilinear interpolation
static float bilinear_interpolate(const Image* img, float x, float y) {
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Check bounds
    if (x0 < 0 || x1 >= img->width || y0 < 0 || y1 >= img->height) {
        return 0.0f;
    }

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img->data[y0 * img->width + x0];
    float v10 = img->data[y0 * img->width + x1];
    float v01 = img->data[y1 * img->width + x0];
    float v11 = img->data[y1 * img->width + x1];

    return (1.0f - dx) * (1.0f - dy) * v00 +
           dx * (1.0f - dy) * v10 +
           (1.0f - dx) * dy * v01 +
           dx * dy * v11;
}

// Core ICA functions
ImageGradients* init_ica(const Image* ref_img, const ICAParams* params) {
    if (!ref_img || !params) return NULL;

    // Create temporary image for gradient computation
    Image* temp_img = create_image(ref_img->height, ref_img->width);
    if (!temp_img) return NULL;

    // Copy image data
    memcpy(temp_img->data, ref_img->data, 
           sizeof(pixel_t) * ref_img->height * ref_img->width);

    // Apply Gaussian blur if needed
    if (params->sigma_blur > 0.0f) {
        apply_gaussian_blur(temp_img, params->sigma_blur);
    }

    // Create gradients structure
    ImageGradients* grads = create_image_gradients(ref_img->height, ref_img->width);
    if (!grads) {
        free_image(temp_img);
        return NULL;
    }

    // Compute gradients
    compute_prewitt_gradients(temp_img, grads);

    // Debug output if enabled
    if (params->debug_mode) {
        printf("Computed gradients with sigma_blur=%.2f\n", params->sigma_blur);
        printf("Gradient range X: [%.2f, %.2f]\n", 
               find_min(grads->data_x, grads->height * grads->width),
               find_max(grads->data_x, grads->height * grads->width));
        printf("Gradient range Y: [%.2f, %.2f]\n",
               find_min(grads->data_y, grads->height * grads->width),
               find_max(grads->data_y, grads->height * grads->width));
    }

    free_image(temp_img);
    return grads;
}

HessianMatrix* compute_hessian(const ImageGradients* grads, int tile_size) {
    int n_patches_y = (grads->height + tile_size - 1) / tile_size;
    int n_patches_x = (grads->width + tile_size - 1) / tile_size;

    HessianMatrix* hessian = create_hessian_matrix(n_patches_y, n_patches_x);
    if (!hessian) return NULL;

    // Compute Hessian for each patch
    for (int py = 0; py < n_patches_y; py++) {
        for (int px = 0; px < n_patches_x; px++) {
            float h00 = 0.0f, h01 = 0.0f, h11 = 0.0f;
            int patch_start_y = py * tile_size;
            int patch_start_x = px * tile_size;

            // Accumulate gradients over patch
            for (int y = 0; y < tile_size; y++) {
                int img_y = patch_start_y + y;
                if (img_y >= grads->height) break;

                for (int x = 0; x < tile_size; x++) {
                    int img_x = patch_start_x + x;
                    if (img_x >= grads->width) break;

                    int idx = img_y * grads->width + img_x;
                    float gx = grads->data_x[idx];
                    float gy = grads->data_y[idx];

                    h00 += gx * gx;
                    h01 += gx * gy;
                    h11 += gy * gy;
                }
            }

            // Store in Hessian matrix
            int hidx = (py * n_patches_x + px) * 4;
            hessian->data[hidx + 0] = h00;  // H[0,0]
            hessian->data[hidx + 1] = h01;  // H[0,1]
            hessian->data[hidx + 2] = h01;  // H[1,0]
            hessian->data[hidx + 3] = h11;  // H[1,1]
        }
    }

    return hessian;
}

void solve_2x2_system(const float* A, const float* b, float* x) {
    float det = A[0] * A[3] - A[1] * A[2];
    if (fabsf(det) < 1e-10f) {
        x[0] = x[1] = 0.0f;
        return;
    }

    float inv_det = 1.0f / det;
    x[0] = (A[3] * b[0] - A[1] * b[1]) * inv_det;
    x[1] = (-A[2] * b[0] + A[0] * b[1]) * inv_det;
}

AlignmentMap* refine_alignment_ica(const Image* ref_img, const Image* alt_img,
                                const ImageGradients* grads,
                                const HessianMatrix* hessian,
                                const AlignmentMap* initial_alignment,
                                const ICAParams* params) {
    // Create a copy of initial alignment to refine
    AlignmentMap* current_alignment = create_alignment_map(initial_alignment->height,
                                                         initial_alignment->width,
                                                         params->tile_size);
    if (!current_alignment) return NULL;

    memcpy(current_alignment->data, initial_alignment->data,
           sizeof(Alignment) * initial_alignment->height * initial_alignment->width);

    // Iterate to refine alignment
    for (int iter = 0; iter < params->num_iterations; iter++) {
        // For each patch
        for (int py = 0; py < current_alignment->height; py++) {
            for (int px = 0; px < current_alignment->width; px++) {
                float b[2] = {0.0f, 0.0f};  // Right-hand side of the system
                int patch_start_y = py * params->tile_size;
                int patch_start_x = px * params->tile_size;
                int hidx = (py * hessian->width + px) * 4;

                // Skip if Hessian is singular
                float det = hessian->data[hidx] * hessian->data[hidx + 3] -
                          hessian->data[hidx + 1] * hessian->data[hidx + 2];
                if (fabsf(det) < 1e-10f) continue;

                // Current alignment for this patch
                Alignment* curr_align = &current_alignment->data[py * current_alignment->width + px];

                // Accumulate gradient differences over patch
                for (int y = 0; y < params->tile_size; y++) {
                    int ref_y = patch_start_y + y;
                    if (ref_y >= ref_img->height) break;

                    for (int x = 0; x < params->tile_size; x++) {
                        int ref_x = patch_start_x + x;
                        if (ref_x >= ref_img->width) break;

                        // Compute warped position
                        float warped_x = ref_x + curr_align->x;
                        float warped_y = ref_y + curr_align->y;

                        // Skip if outside image bounds
                        if (warped_x < 0 || warped_x >= alt_img->width - 1 ||
                            warped_y < 0 || warped_y >= alt_img->height - 1) continue;

                        // Get interpolated value from alternate image
                        float warped_val = bilinear_interpolate(alt_img, warped_x, warped_y);

                        // Compute temporal gradient
                        float ref_val = ref_img->data[ref_y * ref_img->width + ref_x];
                        float dt = warped_val - ref_val;

                        // Update b vector
                        int grad_idx = ref_y * grads->width + ref_x;
                        b[0] += -grads->data_x[grad_idx] * dt;
                        b[1] += -grads->data_y[grad_idx] * dt;
                    }
                }

                // Solve 2x2 system
                float delta[2];
                solve_2x2_system(&hessian->data[hidx], b, delta);

                // Update alignment
                curr_align->x += delta[0];
                curr_align->y += delta[1];
            }
        }
    }

    return current_alignment;
}

// Add this before create_error_map function
static float clamp(float x, float min_val, float max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

ErrorMap* create_error_map(const Image* ref_img, const Image* target_img, 
                          const AlignmentMap* alignment) {
    if (!ref_img || !target_img || !alignment) return NULL;
    
    ErrorMap* error_map = (ErrorMap*)malloc(sizeof(ErrorMap));
    if (!error_map) return NULL;
    
    error_map->height = ref_img->height;
    error_map->width = ref_img->width;
    error_map->data = (float*)malloc(sizeof(float) * ref_img->height * ref_img->width);
    
    if (!error_map->data) {
        free(error_map);
        return NULL;
    }
    
    // Initialize statistics
    float sum_error = 0.0f;
    error_map->min_error = FLT_MAX;
    error_map->max_error = -FLT_MAX;
    
    // Compute error for each pixel
    for (int y = 0; y < ref_img->height; y++) {
        for (int x = 0; x < ref_img->width; x++) {
            // Find which patch this pixel belongs to
            int patch_y = y / alignment->patch_size;
            int patch_x = x / alignment->patch_size;
            
            if (patch_y >= alignment->height) patch_y = alignment->height - 1;
            if (patch_x >= alignment->width) patch_x = alignment->width - 1;
            
            // Get displacement for this patch
            float dx = alignment->data[patch_y * alignment->width + patch_x].x;
            float dy = alignment->data[patch_y * alignment->width + patch_x].y;
            
            // Compute warped position
            float warped_x = x + dx;
            float warped_y = y + dy;
            
            // Get interpolated value from target image
            float warped_val = 0.0f;
            if (warped_x >= 0 && warped_x < target_img->width - 1 &&
                warped_y >= 0 && warped_y < target_img->height - 1) {
                warped_val = bilinear_interpolate(target_img, warped_x, warped_y);
            }
            
            // Compute absolute error
            float ref_val = ref_img->data[y * ref_img->width + x];
            float error = fabsf(warped_val - ref_val);
            
            // Update statistics
            error_map->data[y * ref_img->width + x] = error;
            sum_error += error;
            if (error < error_map->min_error) error_map->min_error = error;
            if (error > error_map->max_error) error_map->max_error = error;
        }
    }
    
    error_map->mean_error = sum_error / (ref_img->height * ref_img->width);
    return error_map;
}

void save_error_map_visualization(const ErrorMap* error_map, const char* filename) {
    if (!error_map || !filename) {
        fprintf(stderr, "Invalid parameters in save_error_map_visualization\n");
        return;
    }
    
    // Create RGB image for visualization
    uint8_t* rgb = (uint8_t*)malloc(error_map->height * error_map->width * 3);
    if (!rgb) {
        fprintf(stderr, "Failed to allocate memory for RGB image\n");
        return;
    }
    
    // Convert error values to colors (using jet colormap)
    for (int y = 0; y < error_map->height; y++) {
        for (int x = 0; x < error_map->width; x++) {
            int idx = y * error_map->width + x;
            float error = error_map->data[idx];
            
            // Normalize error to [0,1]
            float normalized = (error - error_map->min_error) / 
                             (error_map->max_error - error_map->min_error);
            
            // Convert to RGB (jet colormap)
            float r = clamp(1.5f - fabsf(2.0f * normalized - 1.0f), 0.0f, 1.0f);
            float g = clamp(1.5f - fabsf(2.0f * normalized - 0.5f), 0.0f, 1.0f);
            float b = clamp(1.5f - fabsf(2.0f * normalized - 0.0f), 0.0f, 1.0f);
            
            rgb[idx * 3 + 0] = (uint8_t)(r * 255.0f);
            rgb[idx * 3 + 1] = (uint8_t)(g * 255.0f);
            rgb[idx * 3 + 2] = (uint8_t)(b * 255.0f);
        }
    }
    
    // Save as PPM
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        free(rgb);
        return;
    }
    
    fprintf(fp, "P6\n%d %d\n255\n", error_map->width, error_map->height);
    size_t written = fwrite(rgb, 1, error_map->height * error_map->width * 3, fp);
    if (written != error_map->height * error_map->width * 3) {
        fprintf(stderr, "Failed to write complete image data\n");
    }
    fclose(fp);
    
    // Print error statistics
    printf("Error Statistics for %s:\n", filename);
    printf("Min Error: %.6f\n", error_map->min_error);
    printf("Max Error: %.6f\n", error_map->max_error);
    printf("Mean Error: %.6f\n", error_map->mean_error);
    
    free(rgb);
}

void free_error_map(ErrorMap* error_map) {
    if (error_map) {
        free(error_map->data);
        free(error_map);
    }
}


static void print_error_distribution(const ErrorMap* error_map, const char* label) {
    const int num_bins = 10;
    int bins[10] = {0};
    float bin_size = (error_map->max_error - error_map->min_error) / num_bins;
    
    // Count errors in each bin
    for (int i = 0; i < error_map->height * error_map->width; i++) {
        float error = error_map->data[i];
        int bin = (int)((error - error_map->min_error) / bin_size);
        if (bin >= num_bins) bin = num_bins - 1;
        bins[bin]++;
    }
    
    // Print histogram
    printf("\n%s Error Distribution:\n", label);
    int total_pixels = error_map->height * error_map->width;
    for (int i = 0; i < num_bins; i++) {
        float start = error_map->min_error + i * bin_size;
        float end = start + bin_size;
        float percentage = (float)bins[i] / total_pixels * 100.0f;
        printf("%.3f-%.3f: %.1f%% ", start, end, percentage);
        
        // Print simple bar graph
        int bar_length = (int)(percentage / 2);
        for (int j = 0; j < bar_length; j++) printf("*");
        printf("\n");
    }
}


void analyze_error_maps(const ErrorMap* bm_error, const ErrorMap* ica_error) {
    printf("\nError Analysis:\n");
    printf("Block Matching:\n");
    printf("  Mean Error: %.6f\n", bm_error->mean_error);
    printf("  Error Range: [%.6f, %.6f]\n", bm_error->min_error, bm_error->max_error);
    
    printf("\nICA Refinement:\n");
    printf("  Mean Error: %.6f\n", ica_error->mean_error);
    printf("  Error Range: [%.6f, %.6f]\n", ica_error->min_error, ica_error->max_error);
    
    // Calculate improvement
    float mean_improvement = ((bm_error->mean_error - ica_error->mean_error) 
                            / bm_error->mean_error) * 100.0f;
    printf("\nICA Improvement: %.2f%%\n", mean_improvement);
    
    // Print error distribution
    printf("\nError Distribution (in 10 bins):\n");
    print_error_distribution(bm_error, "Block Matching");
    print_error_distribution(ica_error, "ICA Refinement");
}

void analyze_regions(const ErrorMap* error_map, int num_regions_x, int num_regions_y) {
    printf("\nRegion-based Analysis:\n");
    
    int region_width = error_map->width / num_regions_x;
    int region_height = error_map->height / num_regions_y;
    
    // Create heatmap for visualization
    uint8_t* heatmap = (uint8_t*)malloc(error_map->height * error_map->width * 3);
    if (!heatmap) return;
    
    // Analyze each region
    float global_max_error = 0.0f;
    int worst_region_x = 0, worst_region_y = 0;
    
    for (int ry = 0; ry < num_regions_y; ry++) {
        for (int rx = 0; rx < num_regions_x; rx++) {
            RegionStats stats = compute_region_stats(error_map, 
                                                   rx * region_width,
                                                   ry * region_height,
                                                   region_width,
                                                   region_height);
            
            printf("Region (%d,%d):\n", rx, ry);
            printf("  Mean Error: %.6f\n", stats.mean_error);
            printf("  Error Range: [%.6f, %.6f]\n", stats.min_error, stats.max_error);
            
            // Track worst region
            if (stats.mean_error > global_max_error) {
                global_max_error = stats.mean_error;
                worst_region_x = rx;
                worst_region_y = ry;
            }
            
            // Color region in heatmap based on mean error
            float normalized_error = (stats.mean_error - error_map->min_error) / 
                                   (error_map->max_error - error_map->min_error);
            
            // Use jet colormap
            float r = clamp(1.5f - fabsf(2.0f * normalized_error - 1.0f), 0.0f, 1.0f);
            float g = clamp(1.5f - fabsf(2.0f * normalized_error - 0.5f), 0.0f, 1.0f);
            float b = clamp(1.5f - fabsf(2.0f * normalized_error - 0.0f), 0.0f, 1.0f);
            
            // Fill region in heatmap
            for (int y = ry * region_height; y < (ry + 1) * region_height && y < error_map->height; y++) {
                for (int x = rx * region_width; x < (rx + 1) * region_width && x < error_map->width; x++) {
                    int idx = (y * error_map->width + x) * 3;
                    heatmap[idx + 0] = (uint8_t)(r * 255.0f);
                    heatmap[idx + 1] = (uint8_t)(g * 255.0f);
                    heatmap[idx + 2] = (uint8_t)(b * 255.0f);
                }
            }
        }
    }
    
    // Save region heatmap
    FILE* fp = fopen("region_analysis.ppm", "wb");
    if (fp) {
        fprintf(fp, "P6\n%d %d\n255\n", error_map->width, error_map->height);
        fwrite(heatmap, 1, error_map->height * error_map->width * 3, fp);
        fclose(fp);
    }
    
    free(heatmap);
    
    // Print summary
    printf("\nRegion Analysis Summary:\n");
    printf("Worst performing region: (%d,%d)\n", worst_region_x, worst_region_y);
    printf("Maximum regional mean error: %.6f\n", global_max_error);
}

RegionStats compute_region_stats(const ErrorMap* error_map, int x, int y, int width, int height) {
    RegionStats stats = {0};
    stats.x = x;
    stats.y = y;
    stats.width = width;
    stats.height = height;
    stats.min_error = FLT_MAX;
    stats.max_error = -FLT_MAX;
    
    float sum_error = 0.0f;
    int count = 0;
    
    // Compute statistics for region
    for (int ry = y; ry < y + height && ry < error_map->height; ry++) {
        for (int rx = x; rx < x + width && rx < error_map->width; rx++) {
            float error = error_map->data[ry * error_map->width + rx];
            sum_error += error;
            count++;
            
            if (error < stats.min_error) stats.min_error = error;
            if (error > stats.max_error) stats.max_error = error;
        }
    }
    
    stats.mean_error = sum_error / count;
    return stats;
}

Image* create_warped_image(const Image* src_img, const AlignmentMap* alignment) {
    if (!src_img || !alignment) return NULL;
    
    Image* warped = create_image(src_img->height, src_img->width);
    if (!warped) return NULL;
    
    // For each pixel in output image
    for (int y = 0; y < warped->height; y++) {
        for (int x = 0; x < warped->width; x++) {
            // Find which patch this pixel belongs to
            int patch_y = y / alignment->patch_size;
            int patch_x = x / alignment->patch_size;
            
            if (patch_y >= alignment->height) patch_y = alignment->height - 1;
            if (patch_x >= alignment->width) patch_x = alignment->width - 1;
            
            // Get displacement for this patch
            float dx = alignment->data[patch_y * alignment->width + patch_x].x;
            float dy = alignment->data[patch_y * alignment->width + patch_x].y;
            
            // Compute source position
            float src_x = x + dx;
            float src_y = y + dy;
            
            // Get interpolated value from source image
            if (src_x >= 0 && src_x < src_img->width - 1 &&
                src_y >= 0 && src_y < src_img->height - 1) {
                warped->data[y * warped->width + x] = 
                    bilinear_interpolate(src_img, src_x, src_y);
            }
        }
    }
    
    return warped;
}
