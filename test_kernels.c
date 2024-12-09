#include <stdio.h>
#include "kernels.h"

int main() {
    // Create test image
    int width = 64, height = 64;
    Image* test_img = create_image(height, width);
    if (!test_img) {
        fprintf(stderr, "Failed to create test image\n");
        return 1;
    }

    // Fill with test pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a more interesting test pattern
            float dx = x - width/2;
            float dy = y - height/2;
            float r = sqrtf(dx*dx + dy*dy);
            test_img->data[y * width + x] = sinf(r * 0.2f);
        }
    }

    // Set up kernel parameters
    KernelParams params = {
        .type = KERNEL_HANDHELD,
        .k_detail = 1.0f,
        .k_denoise = 0.5f,
        .d_tr = 1.0f,
        .d_th = 0.1f,
        .k_shrink = 2.0f,
        .k_stretch = 4.0f,
        .window_size = 32  // Increased size for better visualization
    };

    // Save input image visualization
    save_kernel_visualization(test_img->data, width, "input_image.ppm");
    printf("Saved input image visualization\n");

    // Estimate kernels
    SteerableKernels* kernels = estimate_kernels(test_img, &params);
    if (kernels) {
        printf("Successfully estimated %d kernels of size %d\n", 
               kernels->count, kernels->size);

        // Visualize kernels and their responses
        visualize_steerable_kernels(kernels, test_img, "steerable");
        printf("Saved kernel and response visualizations\n");

        // Test kernel estimation parameters
        KernelEstimationParams est_params = {
            .k_detail = 1.0f,
            .k_denoise = 0.5f,
            .d_tr = 1.0f,
            .d_th = 0.1f,
            .k_stretch = 4.0f,
            .k_shrink = 2.0f,
            .noise = {.alpha = 0.01f, .beta = 0.001f}
        };

        // Estimate kernel covariances
        int cov_height = height/2;
        int cov_width = width/2;
        Matrix2x2* covs = (Matrix2x2*)malloc(cov_height * cov_width * sizeof(Matrix2x2));
        if (covs) {
            estimate_kernel_covariance(test_img, &est_params, covs, cov_height, cov_width);
            
            // Visualize covariance determinants
            float* det_map = (float*)malloc(cov_height * cov_width * sizeof(float));
            if (det_map) {
                for (int i = 0; i < cov_height * cov_width; i++) {
                    det_map[i] = matrix2x2_determinant(&covs[i]);
                }
                save_kernel_visualization(det_map, cov_width, "covariance_det.ppm");
                free(det_map);
            }
            free(covs);
        }

        free_steerable_kernels(kernels);
    }

    free_image(test_img);
    return 0;
} 