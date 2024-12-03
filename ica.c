#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "ica.h"

// Helper function declarations
static void compute_prewitt_gradients(const Image* img, ImageGradients* grads);
static void apply_gaussian_blur(Image* img, float sigma);
static void compute_gaussian_kernel_1d(float* kernel, int size, float sigma);
static float bilinear_interpolate(const Image* img, float x, float y);

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

    // Compute horizontal gradients
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum_x = 0.0f;
            float sum_y = 0.0f;

            // Apply horizontal kernel
            for (int k = -1; k <= 1; k++) {
                int xk = x + k;
                if (xk >= 0 && xk < img->width) {
                    sum_x += img->data[y * img->width + xk] * kernelx[k + 1];
                }
            }

            // Apply vertical kernel
            for (int k = -1; k <= 1; k++) {
                int yk = y + k;
                if (yk >= 0 && yk < img->height) {
                    sum_y += img->data[yk * img->width + x] * kernely[k + 1];
                }
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

    int kernel_radius = (int)(4.0f * sigma + 0.5f);
    int kernel_size = 2 * kernel_radius + 1;
    float* kernel = (float*)malloc(sizeof(float) * kernel_size);
    float* temp = (float*)malloc(sizeof(float) * img->height * img->width);

    if (!kernel || !temp) {
        free(kernel);
        free(temp);
        return;
    }

    compute_gaussian_kernel_1d(kernel, kernel_size, sigma);

    // Horizontal pass
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;

            for (int k = -kernel_radius; k <= kernel_radius; k++) {
                int xk = x + k;
                if (xk >= 0 && xk < img->width) {
                    float weight = kernel[k + kernel_radius];
                    sum += img->data[y * img->width + xk] * weight;
                    weight_sum += weight;
                }
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
                if (yk >= 0 && yk < img->height) {
                    float weight = kernel[k + kernel_radius];
                    sum += temp[yk * img->width + x] * weight;
                    weight_sum += weight;
                }
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
    // Create temporary image for gradient computation
    Image* temp_img = create_image(ref_img->height, ref_img->width);
    if (!temp_img) return NULL;

    // Copy image data
    memcpy(temp_img->data, ref_img->data, sizeof(pixel_t) * ref_img->height * ref_img->width);

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