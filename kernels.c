#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "kernels.h"
#include "linalg.h"

// Create steerable kernels structure
SteerableKernels* create_steerable_kernels(int size, int count) {
    SteerableKernels* kernels = (SteerableKernels*)malloc(sizeof(SteerableKernels));
    if (!kernels) return NULL;

    kernels->size = size;
    kernels->count = count;
    size_t total_size = size * size * count;

    kernels->weights = (float*)malloc(total_size * sizeof(float));
    kernels->orientations = (float*)malloc(count * sizeof(float));

    if (!kernels->weights || !kernels->orientations) {
        free_steerable_kernels(kernels);
        return NULL;
    }

    return kernels;
}

// Free steerable kernels
void free_steerable_kernels(SteerableKernels* kernels) {
    if (kernels) {
        free(kernels->weights);
        free(kernels->orientations);
        free(kernels);
    }
}

// Compute structure tensor for local orientation estimation
void compute_structure_tensor(const Image* image, float* Ixx, float* Ixy, float* Iyy) {
    int height = image->height;
    int width = image->width;
    size_t size = height * width;
    
    // Allocate temporary gradients
    float* dx = (float*)malloc(size * sizeof(float));
    float* dy = (float*)malloc(size * sizeof(float));
    
    if (!dx || !dy) {
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
    if (!image || !params) return NULL;

    int height = image->height;
    int width = image->width;
    size_t size = height * width;

    // Allocate structure tensor components
    float* Ixx = (float*)malloc(size * sizeof(float));
    float* Ixy = (float*)malloc(size * sizeof(float));
    float* Iyy = (float*)malloc(size * sizeof(float));
    float* magnitudes = (float*)malloc(size * sizeof(float));
    float* orientations = (float*)malloc(size * sizeof(float));

    if (!Ixx || !Ixy || !Iyy || !magnitudes || !orientations) {
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
    free(Ixx);
    free(Ixy);
    free(Iyy);
    free(magnitudes);
    free(orientations);

    return kernels;
} 