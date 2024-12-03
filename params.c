#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "params.h"

// Default parameter values
static const char* DEFAULT_GREY_METHOD = "FFT";
static const int DEFAULT_SCALE = 2;
static const int DEFAULT_TILE_SIZE = 16;
static const float DEFAULT_THRESHOLD = 0.12f;
static const float DEFAULT_S1 = 0.25f;
static const float DEFAULT_S2 = 1.0f;
static const int DEFAULT_MT = 3;
static const float DEFAULT_SIGMA_BLUR = 0.0f;
static const int DEFAULT_KANADE_ITER = 3;
static const float DEFAULT_KERNEL_DETAIL = 1.0f;
static const float DEFAULT_KERNEL_DENOISE = 1.0f;
static const float DEFAULT_D_TR = 0.1f;
static const float DEFAULT_D_TH = 0.1f;
static const float DEFAULT_K_SHRINK = 0.8f;
static const float DEFAULT_K_STRETCH = 1.2f;

// Create default parameters
Params* create_default_params(void) {
    Params* params = (Params*)malloc(sizeof(Params));
    if (!params) return NULL;

    // Initialize with default values
    params->grey_method = strdup(DEFAULT_GREY_METHOD);
    params->scale = DEFAULT_SCALE;
    params->debug_mode = false;

    // Block matching parameters
    params->block_matching.num_levels = 4;
    params->block_matching.tile_size = DEFAULT_TILE_SIZE;
    params->block_matching.search_radius = 4;
    params->block_matching.use_l1_dist = true;

    // ICA parameters
    params->kanade.num_iterations = DEFAULT_KANADE_ITER;
    params->kanade.sigma_blur = DEFAULT_SIGMA_BLUR;
    params->kanade.tile_size = DEFAULT_TILE_SIZE;

    // Robustness parameters
    params->robustness.enabled = true;
    params->robustness.threshold = DEFAULT_THRESHOLD;
    params->robustness.s1 = DEFAULT_S1;
    params->robustness.s2 = DEFAULT_S2;
    params->robustness.mt = DEFAULT_MT;
    params->robustness.window_size = 3;
    params->robustness.epsilon = 1e-6f;

    // Kernel parameters
    params->kernel.type = KERNEL_HANDHELD;
    params->kernel.k_detail = DEFAULT_KERNEL_DETAIL;
    params->kernel.k_denoise = DEFAULT_KERNEL_DENOISE;
    params->kernel.d_tr = DEFAULT_D_TR;
    params->kernel.d_th = DEFAULT_D_TH;
    params->kernel.k_shrink = DEFAULT_K_SHRINK;
    params->kernel.k_stretch = DEFAULT_K_STRETCH;
    params->kernel.window_size = DEFAULT_TILE_SIZE;

    // Merge parameters
    params->merge.power_max = 2.0f;
    params->merge.max_frame_count = 8;
    params->merge.radius_max = 2.0f;
    params->merge.noise_sigma = 0.01f;
    params->merge.use_robustness = true;
    params->merge.use_kernels = true;

    return params;
}

// Free parameters
void free_params(Params* params) {
    if (params) {
        free(params->grey_method);
        free(params);
    }
}

// Validate parameters
bool validate_params(const Params* params, int image_height, int image_width) {
    if (!params) return false;

    // Check grey method
    if (strcmp(params->grey_method, "FFT") != 0) {
        fprintf(stderr, "Error: Grey level images should be obtained with FFT\n");
        return false;
    }

    // Check scale factor
    if (params->scale < 1) {
        fprintf(stderr, "Error: Scale factor must be >= 1\n");
        return false;
    }
    if (params->scale > 3) {
        fprintf(stderr, "Warning: Scale factor > 3 may not produce good results\n");
    }

    // Check block matching parameters
    if (params->block_matching.num_levels < 1 || 
        params->block_matching.num_levels > MAX_PYRAMID_LEVELS) {
        fprintf(stderr, "Error: Invalid number of pyramid levels\n");
        return false;
    }

    if (params->block_matching.tile_size < 4 || 
        params->block_matching.tile_size > 64) {
        fprintf(stderr, "Error: Invalid tile size\n");
        return false;
    }

    // Check ICA parameters
    if (params->kanade.num_iterations < 1) {
        fprintf(stderr, "Error: ICA iterations must be > 0\n");
        return false;
    }

    if (params->kanade.sigma_blur < 0.0f) {
        fprintf(stderr, "Error: Sigma blur must be >= 0\n");
        return false;
    }

    // Check robustness parameters
    if (params->robustness.enabled) {
        if (params->robustness.threshold < 0.0f || 
            params->robustness.threshold > 1.0f) {
            fprintf(stderr, "Error: Robustness threshold must be in [0,1]\n");
            return false;
        }

        if (params->robustness.s1 <= 0.0f || 
            params->robustness.s2 <= 0.0f) {
            fprintf(stderr, "Error: s1 and s2 must be > 0\n");
            return false;
        }
    }

    // Check kernel parameters
    if (params->kernel.k_detail <= 0.0f || 
        params->kernel.k_denoise <= 0.0f) {
        fprintf(stderr, "Error: Kernel parameters must be > 0\n");
        return false;
    }

    // Check merge parameters
    if (params->merge.power_max <= 0.0f || 
        params->merge.radius_max <= 0.0f) {
        fprintf(stderr, "Error: Merge parameters must be > 0\n");
        return false;
    }

    // Check image dimensions
    if (image_height < params->block_matching.tile_size || 
        image_width < params->block_matching.tile_size) {
        fprintf(stderr, "Error: Image dimensions too small for tile size\n");
        return false;
    }

    return true;
}

// Merge parameters from custom settings
void merge_params(Params* base_params, const Params* custom_params) {
    if (!base_params || !custom_params) return;

    // Only override non-NULL values from custom params
    if (custom_params->grey_method) {
        free(base_params->grey_method);
        base_params->grey_method = strdup(custom_params->grey_method);
    }

    if (custom_params->scale > 0) {
        base_params->scale = custom_params->scale;
    }

    // Block matching parameters
    if (custom_params->block_matching.num_levels > 0) {
        base_params->block_matching.num_levels = custom_params->block_matching.num_levels;
    }
    if (custom_params->block_matching.tile_size > 0) {
        base_params->block_matching.tile_size = custom_params->block_matching.tile_size;
    }
    if (custom_params->block_matching.search_radius > 0) {
        base_params->block_matching.search_radius = custom_params->block_matching.search_radius;
    }

    // ICA parameters
    if (custom_params->kanade.num_iterations > 0) {
        base_params->kanade.num_iterations = custom_params->kanade.num_iterations;
    }
    if (custom_params->kanade.sigma_blur >= 0.0f) {
        base_params->kanade.sigma_blur = custom_params->kanade.sigma_blur;
    }

    // Robustness parameters
    base_params->robustness.enabled = custom_params->robustness.enabled;
    if (custom_params->robustness.threshold > 0.0f) {
        base_params->robustness.threshold = custom_params->robustness.threshold;
    }
    if (custom_params->robustness.s1 > 0.0f) {
        base_params->robustness.s1 = custom_params->robustness.s1;
    }
    if (custom_params->robustness.s2 > 0.0f) {
        base_params->robustness.s2 = custom_params->robustness.s2;
    }

    // Kernel parameters
    base_params->kernel.type = custom_params->kernel.type;
    if (custom_params->kernel.k_detail > 0.0f) {
        base_params->kernel.k_detail = custom_params->kernel.k_detail;
    }
    if (custom_params->kernel.k_denoise > 0.0f) {
        base_params->kernel.k_denoise = custom_params->kernel.k_denoise;
    }

    // Merge parameters
    if (custom_params->merge.power_max > 0.0f) {
        base_params->merge.power_max = custom_params->merge.power_max;
    }
    if (custom_params->merge.max_frame_count > 0) {
        base_params->merge.max_frame_count = custom_params->merge.max_frame_count;
    }
    if (custom_params->merge.radius_max > 0.0f) {
        base_params->merge.radius_max = custom_params->merge.radius_max;
    }

    // Update dependent parameters
    if (base_params->block_matching.tile_size != DEFAULT_TILE_SIZE) {
        base_params->kanade.tile_size = base_params->block_matching.tile_size;
        base_params->kernel.window_size = base_params->block_matching.tile_size;
    }
}

// Print parameters for debugging
void print_params(const Params* params) {
    if (!params) return;

    printf("Parameters:\n");
    printf("  Grey Method: %s\n", params->grey_method);
    printf("  Scale: %d\n", params->scale);
    printf("  Debug Mode: %s\n", params->debug_mode ? "true" : "false");

    printf("\nBlock Matching:\n");
    printf("  Levels: %d\n", params->block_matching.num_levels);
    printf("  Tile Size: %d\n", params->block_matching.tile_size);
    printf("  Search Radius: %d\n", params->block_matching.search_radius);

    printf("\nICA:\n");
    printf("  Iterations: %d\n", params->kanade.num_iterations);
    printf("  Sigma Blur: %.3f\n", params->kanade.sigma_blur);

    printf("\nRobustness:\n");
    printf("  Enabled: %s\n", params->robustness.enabled ? "true" : "false");
    printf("  Threshold: %.3f\n", params->robustness.threshold);
    printf("  S1: %.3f\n", params->robustness.s1);
    printf("  S2: %.3f\n", params->robustness.s2);

    printf("\nKernel:\n");
    printf("  Type: %s\n", params->kernel.type == KERNEL_HANDHELD ? "handheld" : "iso");
    printf("  Detail: %.3f\n", params->kernel.k_detail);
    printf("  Denoise: %.3f\n", params->kernel.k_denoise);

    printf("\nMerge:\n");
    printf("  Power Max: %.3f\n", params->merge.power_max);
    printf("  Max Frames: %d\n", params->merge.max_frame_count);
    printf("  Radius Max: %.3f\n", params->merge.radius_max);
    printf("  Use Robustness: %s\n", params->merge.use_robustness ? "true" : "false");
    printf("  Use Kernels: %s\n", params->merge.use_kernels ? "true" : "false");
} 