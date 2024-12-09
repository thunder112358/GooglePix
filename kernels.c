#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "kernels.h"
#include "linalg.h"

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Add debug logging macro
#ifdef DEBUG_KERNELS
#define LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

// Add error checking helper
static bool check_allocation(void* ptr, const char* name) {
    if (!ptr) {
        fprintf(stderr, "Failed to allocate memory for %s\n", name);
        return false;
    }
    return true;
}

// Add input validation
static bool validate_kernel_params(const KernelEstimationParams* params) {
    if (!params) {
        fprintf(stderr, "Null kernel parameters\n");
        return false;
    }
    
    if (params->k_detail < 0.0f || params->k_denoise < 0.0f ||
        params->d_tr <= 0.0f || params->k_shrink <= 0.0f ||
        params->k_stretch <= 0.0f) {
        fprintf(stderr, "Invalid kernel parameters\n");
        return false;
    }
    
    return true;
}

// Compute structure tensor for local orientation estimation
void compute_structure_tensor(const Image* image, float* Ixx, float* Ixy, float* Iyy) {
    if (!image || !Ixx || !Ixy || !Iyy) {
        fprintf(stderr, "Invalid input to compute_structure_tensor\n");
        return;
    }

    int height = image->height;
    int width = image->width;
    size_t size = height * width;
    
    // Allocate temporary gradients with error checking
    float* dx = (float*)malloc(size * sizeof(float));
    float* dy = (float*)malloc(size * sizeof(float));
    
    if (!check_allocation(dx, "dx") || !check_allocation(dy, "dy")) {
        free(dx);
        free(dy);
        return;
    }

    // Compute gradients
    compute_gradient_x(image->data, dx, height, width);
    compute_gradient_y(image->data, dy, height, width);

    // Compute tensor components
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float gx = dx[idx];
            float gy = dy[idx];
            
            Ixx[idx] = gx * gx;
            Ixy[idx] = gx * gy;
            Iyy[idx] = gy * gy;
        }
    }

    // Apply Gaussian smoothing to tensor components
    float sigma = 1.5f;  // Smoothing parameter
    gaussian_blur(Ixx, height, width, sigma);
    gaussian_blur(Ixy, height, width, sigma);
    gaussian_blur(Iyy, height, width, sigma);

    free(dx);
    free(dy);
}

// Estimate local orientation from structure tensor
void estimate_local_orientation(const float* Ixx, const float* Ixy, const float* Iyy,
                              float* magnitudes, float* orientations,
                              int height, int width) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float xx = Ixx[idx];
            float xy = Ixy[idx];
            float yy = Iyy[idx];

            // Compute eigenvalues
            float a = xx + yy;
            float b = sqrtf((xx - yy) * (xx - yy) + 4 * xy * xy);
            float lambda1 = 0.5f * (a + b);
            float lambda2 = 0.5f * (a - b);

            // Compute orientation (perpendicular to dominant eigenvector)
            orientations[idx] = 0.5f * atan2f(2.0f * xy, xx - yy);
            
            // Compute anisotropy measure
            magnitudes[idx] = compute_anisotropy(lambda1, lambda2);
        }
    }
}

// Generate steerable basis kernel for given orientation
void generate_steerable_basis(float* basis, int size, float orientation) {
    int radius = size / 2;
    float cos_theta = cosf(orientation);
    float sin_theta = sinf(orientation);
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = (float)(x - radius);
            float dy = (float)(y - radius);
            
            // Rotate coordinates
            float rx = dx * cos_theta + dy * sin_theta;
            float ry = -dx * sin_theta + dy * cos_theta;
            
            // Compute anisotropic Gaussian
            float sigma_x = 1.0f;
            float sigma_y = 0.5f;
            float norm = 1.0f / (2.0f * M_PI * sigma_x * sigma_y);
            
            basis[y * size + x] = norm * expf(-0.5f * (
                (rx * rx) / (sigma_x * sigma_x) +
                (ry * ry) / (sigma_y * sigma_y)
            ));
        }
    }
}

// Compute anisotropy measure from eigenvalues
float compute_anisotropy(float lambda1, float lambda2) {
    const float epsilon = 1e-6f;
    float sum = lambda1 + lambda2 + epsilon;
    return (lambda1 - lambda2) / sum;
}

// Main kernel estimation function
SteerableKernels* estimate_kernels(const Image* image, const KernelParams* params) {
    if (!image || !params) {
        fprintf(stderr, "Invalid input to estimate_kernels\n");
        return NULL;
    }

    int height = image->height;
    int width = image->width;
    size_t size = height * width;

    // Allocate structure tensor components with error checking
    float* Ixx = (float*)malloc(size * sizeof(float));
    float* Ixy = (float*)malloc(size * sizeof(float));
    float* Iyy = (float*)malloc(size * sizeof(float));
    float* magnitudes = (float*)malloc(size * sizeof(float));
    float* orientations = (float*)malloc(size * sizeof(float));

    if (!check_allocation(Ixx, "Ixx") || !check_allocation(Ixy, "Ixy") ||
        !check_allocation(Iyy, "Iyy") || !check_allocation(magnitudes, "magnitudes") ||
        !check_allocation(orientations, "orientations")) {
        free(Ixx); free(Ixy); free(Iyy);
        free(magnitudes); free(orientations);
        return NULL;
    }

    // Compute structure tensor
    compute_structure_tensor(image, Ixx, Ixy, Iyy);
    
    // Estimate local orientations
    estimate_local_orientation(Ixx, Ixy, Iyy, magnitudes, orientations, height, width);

    // Create steerable kernels
    int kernel_count = (params->type == KERNEL_HANDHELD) ? 8 : 1;
    SteerableKernels* kernels = create_steerable_kernels(params->window_size, kernel_count);
    
    if (kernels) {
        if (params->type == KERNEL_HANDHELD) {
            // Generate oriented kernels
            for (int i = 0; i < kernel_count; i++) {
                float angle = (float)i * M_PI / kernel_count;
                kernels->orientations[i] = angle;
                generate_steerable_basis(
                    kernels->weights + i * params->window_size * params->window_size,
                    params->window_size, angle
                );
            }
        } else {
            // Generate isotropic kernel
            kernels->orientations[0] = 0.0f;
            generate_steerable_basis(
                kernels->weights,
                params->window_size, 0.0f
            );
        }
    }

    // Cleanup
    free(Ixx); free(Ixy); free(Iyy);
    free(magnitudes); free(orientations);

    return kernels;
}

// Add GAT implementation
void apply_gat(float* image, int size, const NoiseParams* noise) {
    if (!image || !noise || size <= 0) {
        fprintf(stderr, "Invalid input to apply_gat\n");
        return;
    }
    
    LOG_DEBUG("Applying GAT with alpha=%.3f, beta=%.3f", noise->alpha, noise->beta);
    
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        float x = image[i];
        float sigma2 = noise->alpha * x + noise->beta;
        if (sigma2 < 0.0f) {
            LOG_DEBUG("Warning: negative sigma2 at index %d", i);
            sigma2 = 0.0f;
        }
        image[i] = 2.0f * sqrtf(fmaxf(x + 3.0f/8.0f + sigma2, 0.0f));
    }
}

// Add Prewitt gradient computation
void compute_gradients_prewitt(const Image* image, float* grad_x, float* grad_y) {
    if (!image || !grad_x || !grad_y || !image->data) {
        fprintf(stderr, "Invalid input to compute_gradients_prewitt\n");
        return;
    }
    
    if (image->width < 2 || image->height < 2) {
        fprintf(stderr, "Image too small for gradient computation\n");
        return;
    }
    
    LOG_DEBUG("Computing Prewitt gradients for %dx%d image", image->height, image->width);
    
    // Prewitt kernels matching Python implementation
    const float kernel_x[2][2] = {{-0.5f, 0.5f}, {0.5f, 0.5f}};
    const float kernel_y[2][2] = {{0.5f, 0.5f}, {-0.5f, 0.5f}};
    
    int height = image->height;
    int width = image->width;
    
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height-1; y++) {
        for (int x = 0; x < width-1; x++) {
            float gx = 0.0f, gy = 0.0f;
            
            // Apply kernels
            for (int ky = 0; ky < 2; ky++) {
                for (int kx = 0; kx < 2; kx++) {
                    float val = image->data[(y+ky) * width + (x+kx)];
                    gx += val * kernel_x[ky][kx];
                    gy += val * kernel_y[ky][kx];
                }
            }
            
            grad_x[y * width + x] = gx;
            grad_y[y * width + x] = gy;
        }
    }
}

// Add kernel estimation matching Python implementation
void estimate_kernel_covariance(const Image* image, 
                              const KernelEstimationParams* params,
                              Matrix2x2* covs, int cov_height, int cov_width) {
    if (!image || !params || !covs) {
        fprintf(stderr, "Invalid input to estimate_kernel_covariance\n");
        return;
    }
    
    if (!validate_kernel_params(params)) {
        return;
    }
    
    if (cov_height <= 0 || cov_width <= 0) {
        fprintf(stderr, "Invalid covariance dimensions: %dx%d\n", cov_height, cov_width);
        return;
    }
    
    LOG_DEBUG("Estimating kernels for %dx%d image, output size %dx%d",
              image->height, image->width, cov_height, cov_width);
    
    // Allocate gradients with error checking
    float* grad_x = (float*)malloc(image->width * image->height * sizeof(float));
    float* grad_y = (float*)malloc(image->width * image->height * sizeof(float));
    
    if (!check_allocation(grad_x, "grad_x") || !check_allocation(grad_y, "grad_y")) {
        free(grad_x);
        free(grad_y);
        return;
    }
    
    // Apply GAT and compute gradients
    apply_gat(image->data, image->width * image->height, &params->noise);
    compute_gradients_prewitt(image, grad_x, grad_y);
    
    // Process each position
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < cov_height; y++) {
        for (int x = 0; x < cov_width; x++) {
            Matrix2x2 structure_tensor = {0};
            
            // Accumulate structure tensor
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    int img_y = y*2 - 1 + i;
                    int img_x = x*2 - 1 + j;
                    
                    if (img_y >= 0 && img_y < image->height &&
                        img_x >= 0 && img_x < image->width) {
                        float gx = grad_x[img_y * image->width + img_x];
                        float gy = grad_y[img_y * image->width + img_x];
                        
                        structure_tensor.a11 += gx * gx;
                        structure_tensor.a12 += gx * gy;
                        structure_tensor.a21 += gx * gy;
                        structure_tensor.a22 += gy * gy;
                    }
                }
            }
            
            // Compute eigendecomposition
            float eigenvals[2];
            Vector2D e1, e2;
            get_eigen_elmts_2x2(&structure_tensor, eigenvals, &e1, &e2);
            
            // Validate eigenvalues
            if (eigenvals[0] < 0.0f || eigenvals[1] < 0.0f) {
                LOG_DEBUG("Warning: negative eigenvalues at (%d,%d): %.3f, %.3f",
                         x, y, eigenvals[0], eigenvals[1]);
                eigenvals[0] = fabsf(eigenvals[0]);
                eigenvals[1] = fabsf(eigenvals[1]);
            }
            
            // Compute kernel parameters
            float sum_eigenvals = eigenvals[0] + eigenvals[1] + 1e-6f;
            float A = 1.0f + sqrtf((eigenvals[0] - eigenvals[1]) / sum_eigenvals);
            float D = fmaxf(1.0f - sqrtf(eigenvals[0])/params->d_tr + params->d_th, 0.0f);
            
            // Compute k values
            float k1, k2;
            if (A > 1.95f) {
                k1 = 1.0f/params->k_shrink;
                k2 = params->k_stretch;
            } else {
                k1 = k2 = 1.0f;
            }
            
            k1 = params->k_detail * ((1.0f-D)*k1 + D*params->k_denoise);
            k2 = params->k_detail * ((1.0f-D)*k2 + D*params->k_denoise);
            
            // Store result
            int idx = y * cov_width + x;
            covs[idx].a11 = k1*k1*e1.x*e1.x + k2*k2*e2.x*e2.x;
            covs[idx].a12 = k1*k1*e1.x*e1.y + k2*k2*e2.x*e2.y;
            covs[idx].a21 = covs[idx].a12;
            covs[idx].a22 = k1*k1*e1.y*e1.y + k2*k2*e2.y*e2.y;
            
            LOG_DEBUG("Kernel at (%d,%d): A=%.3f, D=%.3f, k1=%.3f, k2=%.3f",
                     x, y, A, D, k1, k2);
        }
    }
    
    free(grad_x);
    free(grad_y);
}

// Add visualization functions
void save_kernel_visualization(const float* kernel, int size, const char* filename) {
    // Find min/max for normalization
    float min_val = kernel[0], max_val = kernel[0];
    for (int i = 0; i < size * size; i++) {
        if (kernel[i] < min_val) min_val = kernel[i];
        if (kernel[i] > max_val) max_val = kernel[i];
    }
    float range = max_val - min_val;
    if (range < 1e-6f) range = 1.0f;

    // Open file
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", size, size);

    // Write RGB data
    uint8_t* rgb = (uint8_t*)malloc(size * size * 3);
    if (rgb) {
        for (int i = 0; i < size * size; i++) {
            // Normalize to [0,1]
            float val = (kernel[i] - min_val) / range;
            
            // Use jet colormap
            float r = clamp(1.5f - fabsf(2.0f * val - 1.0f), 0.0f, 1.0f);
            float g = clamp(1.5f - fabsf(2.0f * val - 0.5f), 0.0f, 1.0f);
            float b = clamp(1.5f - fabsf(2.0f * val - 0.0f), 0.0f, 1.0f);
            
            rgb[i*3 + 0] = (uint8_t)(r * 255.0f);
            rgb[i*3 + 1] = (uint8_t)(g * 255.0f);
            rgb[i*3 + 2] = (uint8_t)(b * 255.0f);
        }
        fwrite(rgb, 1, size * size * 3, fp);
        free(rgb);
    }
    fclose(fp);
}

// Add kernel response computation
KernelResponse* compute_kernel_response(const Image* image, const float* kernel, int kernel_size) {
    if (!image || !kernel || kernel_size <= 0) {
        fprintf(stderr, "Invalid input to compute_kernel_response\n");
        return NULL;
    }

    KernelResponse* response = (KernelResponse*)malloc(sizeof(KernelResponse));
    if (!response) return NULL;

    response->height = image->height - kernel_size + 1;
    response->width = image->width - kernel_size + 1;
    response->response = (float*)malloc(response->height * response->width * sizeof(float));

    if (!response->response) {
        free(response);
        return NULL;
    }

    // Compute response using correlation
    response->min_val = FLT_MAX;
    response->max_val = -FLT_MAX;

    int half_size = kernel_size / 2;

    #pragma omp parallel for collapse(2)
    for (int y = half_size; y < image->height - half_size; y++) {
        for (int x = half_size; x < image->width - half_size; x++) {
            float sum = 0.0f;
            
            // Apply kernel
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int img_y = y + ky - half_size;
                    int img_x = x + kx - half_size;
                    sum += image->data[img_y * image->width + img_x] * 
                           kernel[ky * kernel_size + kx];
                }
            }

            int resp_y = y - half_size;
            int resp_x = x - half_size;
            response->response[resp_y * response->width + resp_x] = sum;

            #pragma omp critical
            {
                if (sum < response->min_val) response->min_val = sum;
                if (sum > response->max_val) response->max_val = sum;
            }
        }
    }

    return response;
}

void save_response_visualization(const KernelResponse* response, const char* filename) {
    if (!response || !filename) return;

    save_kernel_visualization(response->response, response->width, filename);
}

void free_kernel_response(KernelResponse* response) {
    if (response) {
        free(response->response);
        free(response);
    }
}

// Update visualize_steerable_kernels to include response visualization
void visualize_steerable_kernels(const SteerableKernels* kernels, const Image* image, const char* prefix) {
    if (!kernels || !prefix || !image) {
        fprintf(stderr, "Invalid input to visualize_steerable_kernels\n");
        return;
    }

    char filename[256];
    for (int i = 0; i < kernels->count; i++) {
        // Save kernel visualization
        snprintf(filename, sizeof(filename), "%s_kernel_%02d.ppm", prefix, i);
        save_kernel_visualization(
            kernels->weights + i * kernels->size * kernels->size,
            kernels->size,
            filename
        );
        
        // Compute and save response visualization
        KernelResponse* response = compute_kernel_response(
            image,
            kernels->weights + i * kernels->size * kernels->size,
            kernels->size
        );

        if (response) {
            snprintf(filename, sizeof(filename), "%s_response_%02d.ppm", prefix, i);
            save_response_visualization(response, filename);
            
            printf("Kernel %d: orientation = %.2f degrees, response range = [%.3f, %.3f]\n",
                   i, kernels->orientations[i] * 180.0f / M_PI,
                   response->min_val, response->max_val);
            
            free_kernel_response(response);
        }
    }
} 