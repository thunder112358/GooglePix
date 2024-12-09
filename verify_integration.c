#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "block_matching.h"
#include "ica.h"
#include "kernels.h"

// Helper function to create test image
static Image* create_test_image(int height, int width, int channels) {
    Image* img = create_image_channels(height, width, channels);
    if (!img) return NULL;
    
    // Create test pattern (sine wave with some features)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = x - width/2;
            float dy = y - height/2;
            float r = sqrtf(dx*dx + dy*dy);
            float angle = atan2f(dy, dx);
            // Create pattern with both radial and angular components
            float val = (sinf(r * 0.2f) + cosf(angle * 4)) * 0.25f + 0.5f;
            
            for (int c = 0; c < channels; c++) {
                img->data[(y * width + x) * channels + c] = val;
            }
        }
    }
    
    return img;
}

// Helper function to save image as PPM
static void save_image_ppm(const Image* img, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);
    
    unsigned char* rgb = (unsigned char*)malloc(img->width * img->height * 3);
    if (rgb) {
        for (int i = 0; i < img->width * img->height; i++) {
            for (int c = 0; c < 3; c++) {
                int idx = i * img->channels + (c % img->channels);
                rgb[i * 3 + c] = (unsigned char)(img->data[idx] * 255.0f);
            }
        }
        fwrite(rgb, 1, img->width * img->height * 3, fp);
        free(rgb);
    }
    
    fclose(fp);
}

int main() {
    // Test parameters
    int width = 256;
    int height = 256;
    int channels = 1;  // Start with grayscale
    
    // Create test images
    Image* ref_img = create_test_image(height, width, channels);
    Image* target_img = create_test_image(height, width, channels);
    if (!ref_img || !target_img) {
        printf("Failed to create test images\n");
        return 1;
    }
    
    // Add artificial displacement to target image
    float shift_x = 5.0f;
    float shift_y = 3.0f;
    Image* shifted_target = create_image_channels(height, width, channels);
    if (shifted_target) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_x = x - (int)shift_x;
                int src_y = y - (int)shift_y;
                if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                    shifted_target->data[y * width + x] = target_img->data[src_y * width + src_x];
                }
            }
        }
    }
    
    // Save input images
    save_image_ppm(ref_img, "test_ref.ppm");
    save_image_ppm(shifted_target, "test_target.ppm");
    
    printf("Step 1: Block Matching\n");
    
    // Setup block matching parameters
    BlockMatchingParams bm_params = {
        .num_levels = 3,
        .factors = (int[]){1, 2, 4},
        .tile_sizes = (int[]){16, 32, 64},
        .search_radii = (int[]){8, 4, 4},
        .use_l1_dist = (bool[]){true, true, true},
        .pad_top = 0,
        .pad_bottom = 0,
        .pad_left = 0,
        .pad_right = 0
    };
    
    // Initialize pyramid
    ImagePyramid* ref_pyramid = init_block_matching(ref_img, &bm_params);
    if (!ref_pyramid) {
        printf("Failed to initialize block matching\n");
        return 1;
    }
    
    // Perform block matching
    AlignmentMap* alignment = align_image_block_matching(shifted_target, ref_pyramid, &bm_params);
    if (!alignment) {
        printf("Block matching failed\n");
        return 1;
    }
    
    printf("Step 2: ICA Refinement\n");
    
    // Setup ICA parameters
    ICAParams ica_params = {
        .sigma_blur = 1.0f,
        .num_iterations = 5,
        .tile_size = 16,
        .debug_mode = true,
        .save_iterations = false,
        .debug_dir = ".",
        .convergence_threshold = 0.1f
    };
    
    // Initialize ICA
    ImageGradients* grads = init_ica(ref_img, &ica_params);
    if (!grads) {
        printf("Failed to initialize ICA\n");
        return 1;
    }
    
    // Compute Hessian
    HessianMatrix* hessian = compute_hessian(grads, ica_params.tile_size);
    if (!hessian) {
        printf("Failed to compute Hessian\n");
        return 1;
    }
    
    // Refine alignment using ICA
    AlignmentMap* refined_alignment = refine_alignment_ica(ref_img, shifted_target, grads, 
                                                         hessian, alignment, &ica_params);
    if (!refined_alignment) {
        printf("ICA refinement failed\n");
        return 1;
    }
    
    printf("Step 3: Kernel Estimation\n");
    
    // Setup kernel parameters
    KernelParams kernel_params = {
        .type = KERNEL_HANDHELD,
        .k_detail = 1.0f,
        .k_denoise = 0.5f,
        .d_tr = 1.0f,
        .d_th = 0.1f,
        .k_shrink = 2.0f,
        .k_stretch = 4.0f,
        .window_size = 32
    };
    
    // Estimate kernels
    SteerableKernels* kernels = estimate_kernels(ref_img, &kernel_params);
    if (!kernels) {
        printf("Kernel estimation failed\n");
        return 1;
    }
    
    // Save visualizations
    save_kernel_visualization(kernels->weights, kernels->size, "kernel_0.ppm");
    
    // Cleanup
    free_steerable_kernels(kernels);
    free_alignment_map(refined_alignment);
    free_alignment_map(alignment);
    free_hessian_matrix(hessian);
    free_image_gradients(grads);
    free_image_pyramid(ref_pyramid);
    free_image(ref_img);
    free_image(target_img);
    free_image(shifted_target);
    
    printf("Test completed successfully\n");
    return 0;
} 