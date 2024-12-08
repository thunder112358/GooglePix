#include <stdio.h>
#include <stdlib.h>
#include "ica.h"

// Function to load test data from NPZ file
static int load_test_data(const char* filename, Image** ref_img, Image** target_img,
                         ImageGradients** grads, AlignmentMap** initial_alignment) {
    // Load NPZ file using a simple NPZ reader (implementation needed)
    // For now, we'll create synthetic data
    int height = 64, width = 64;
    
    // Create reference image
    *ref_img = create_image(height, width);
    if (!*ref_img) return 0;
    
    // Create target image
    *target_img = create_image(height, width);
    if (!*target_img) {
        free_image(*ref_img);
        return 0;
    }
    
    // Fill with test pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = (float)(x + y) / (width + height);
            (*ref_img)->data[y * width + x] = val;
            (*target_img)->data[y * width + x] = val;
        }
    }
    
    // Create gradients
    *grads = create_image_gradients(height, width);
    if (!*grads) {
        free_image(*ref_img);
        free_image(*target_img);
        return 0;
    }
    
    // Create initial alignment
    int patch_size = 8;
    *initial_alignment = create_alignment_map(height/patch_size, width/patch_size, patch_size);
    if (!*initial_alignment) {
        free_image(*ref_img);
        free_image(*target_img);
        free_image_gradients(*grads);
        return 0;
    }
    
    return 1;
}

int main() {
    Image *ref_img = NULL, *target_img = NULL;
    ImageGradients *grads = NULL;
    AlignmentMap *initial_alignment = NULL;
    
    // Load test data
    if (!load_test_data("ica_test_data.npz", &ref_img, &target_img, &grads, &initial_alignment)) {
        fprintf(stderr, "Failed to load test data\n");
        return 1;
    }
    
    // Setup ICA parameters
    ICAParams params = {
        .sigma_blur = 0.5f,
        .num_iterations = 5,
        .tile_size = 8
    };
    
    // Initialize ICA
    grads = init_ica(ref_img, &params);
    if (!grads) {
        fprintf(stderr, "Failed to initialize ICA\n");
        goto cleanup;
    }
    
    // Compute Hessian
    HessianMatrix* hessian = compute_hessian(grads, params.tile_size);
    if (!hessian) {
        fprintf(stderr, "Failed to compute Hessian\n");
        goto cleanup;
    }
    
    // Refine alignment
    AlignmentMap* refined_alignment = refine_alignment_ica(ref_img, target_img, grads, 
                                                         hessian, initial_alignment, &params);
    if (!refined_alignment) {
        fprintf(stderr, "Failed to refine alignment\n");
        free_hessian_matrix(hessian);
        goto cleanup;
    }
    
    // Print results
    printf("Refined Alignment Results:\n");
    for (int y = 0; y < refined_alignment->height; y++) {
        for (int x = 0; x < refined_alignment->width; x++) {
            printf("Patch (%d,%d): dx=%.1f dy=%.1f\n",
                   y, x,
                   refined_alignment->data[y * refined_alignment->width + x].x,
                   refined_alignment->data[y * refined_alignment->width + x].y);
        }
    }
    
    // Cleanup
    free_alignment_map(refined_alignment);
    free_hessian_matrix(hessian);
    
cleanup:
    free_image(ref_img);
    free_image(target_img);
    free_image_gradients(grads);
    free_alignment_map(initial_alignment);
    
    return 0;
} 