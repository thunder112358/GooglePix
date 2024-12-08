#include <stdio.h>
#include "block_matching.h"

// Test function to compare C and Python implementations
void test_block_matching() {
    // Create test images
    int height = 64;
    int width = 64;
    Image* ref_img = create_image(height, width);
    Image* target_img = create_image(height, width);

    // Fill with test pattern
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Reference image: diagonal gradient
            ref_img->data[y * width + x] = (float)(x + y) / (width + height);
            
            // Target image: shifted diagonal gradient
            int shift_x = 2;
            int shift_y = 3;
            int tx = (x + shift_x) % width;
            int ty = (y + shift_y) % height;
            target_img->data[y * width + x] = (float)(tx + ty) / (width + height);
        }
    }

    // Setup block matching parameters
    BlockMatchingParams params = {
        .num_levels = 3,
        .factors = (int[]){1, 2, 4},
        .tile_sizes = (int[]){8, 16, 32},
        .search_radii = (int[]){4, 4, 4},
        .use_l1_dist = (bool[]){true, true, true},
        .pad_top = 0,
        .pad_bottom = 0,
        .pad_left = 0,
        .pad_right = 0,
        .debug_mode = true
    };

    // Initialize pyramid
    ImagePyramid* ref_pyramid = init_block_matching(ref_img, &params);
    
    // Perform block matching
    AlignmentMap* alignments = align_image_block_matching(target_img, ref_pyramid, &params);

    // Print results
    printf("Alignment Results:\n");
    for (int y = 0; y < alignments->height; y++) {
        for (int x = 0; x < alignments->width; x++) {
            printf("Patch (%d,%d): dx=%.1f dy=%.1f\n", 
                   y, x,
                   alignments->data[y * alignments->width + x].x,
                   alignments->data[y * alignments->width + x].y);
        }
    }

    // Cleanup
    free_image(ref_img);
    free_image(target_img);
    free_image_pyramid(ref_pyramid);
    free_alignment_map(alignments);
}

int main() {
    test_block_matching();
    return 0;
} 