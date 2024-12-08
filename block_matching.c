#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "block_matching.h"

// Helper function declarations
static void compute_gaussian_kernel(float* kernel, int size, float sigma);
static void apply_gaussian_blur(Image* img, float sigma);
static void compute_gaussian_kernel_1d(float* kernel, int radius, float sigma);
static Image* apply_gaussian_filter(const Image* img, float sigma);
static void refine_alignment_with_candidates(const Image* ref_img, const Image* target_img,
                                           int patch_y, int patch_x, int patch_size,
                                           const AlignmentMap* prev_alignment,
                                           float scale_factor, int prev_width, int prev_height,
                                           float* best_x, float* best_y,
                                           const BlockMatchingParams* params);
static void save_refinement_visualization(const RefinementDebugInfo* debug_info, const char* output_dir);

// Memory management functions
Image* create_image_channels(int height, int width, int channels) {
    Image* img = (Image*)malloc(sizeof(Image));
    if (!img) return NULL;

    img->height = height;
    img->width = width;
    img->channels = channels;
    img->data = (float*)calloc(height * width * channels, sizeof(float));

    if (!img->data) {
        free(img);
        return NULL;
    }

    return img;
}

Image* create_image(int height, int width) {
    return create_image_channels(height, width, 3);  // Default to 3 channels (RGB)
}

void free_image(Image* img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

AlignmentMap* create_alignment_map(int height, int width, int patch_size) {
    AlignmentMap* map = (AlignmentMap*)malloc(sizeof(AlignmentMap));
    if (!map) return NULL;

    map->height = height;
    map->width = width;
    map->patch_size = patch_size;
    map->data = (Alignment*)calloc(height * width, sizeof(Alignment));
    
    if (!map->data) {
        free(map);
        return NULL;
    }

    return map;
}

void free_alignment_map(AlignmentMap* map) {
    if (map) {
        free(map->data);
        free(map);
    }
}

ImagePyramid* create_image_pyramid(int num_levels) {
    ImagePyramid* pyramid = (ImagePyramid*)malloc(sizeof(ImagePyramid));
    if (!pyramid) return NULL;

    pyramid->num_levels = num_levels;
    pyramid->levels = (Image**)calloc(num_levels, sizeof(Image*));
    pyramid->factors = (int*)malloc(sizeof(int) * num_levels);
    
    if (!pyramid->levels || !pyramid->factors) {
        free(pyramid->levels);
        free(pyramid->factors);
        free(pyramid);
        return NULL;
    }

    return pyramid;
}

void free_image_pyramid(ImagePyramid* pyramid) {
    if (pyramid) {
        for (int i = 0; i < pyramid->num_levels; i++) {
            free_image(pyramid->levels[i]);
        }
        free(pyramid->levels);
        free(pyramid->factors);
        free(pyramid);
    }
}

// Image processing functions
Image* pad_image(const Image* img, int pad_top, int pad_bottom, int pad_left, int pad_right) {
    int new_height = img->height + pad_top + pad_bottom;
    int new_width = img->width + pad_left + pad_right;
    
    Image* padded = create_image(new_height, new_width);
    if (!padded) return NULL;

    // Copy original image
    for (int y = 0; y < img->height; y++) {
        memcpy(&padded->data[(y + pad_top) * new_width + pad_left],
               &img->data[y * img->width],
               img->width * sizeof(pixel_t));
    }

    // Pad top and bottom
    for (int y = 0; y < pad_top; y++) {
        memcpy(&padded->data[y * new_width + pad_left],
               &img->data[0],
               img->width * sizeof(pixel_t));
    }
    for (int y = 0; y < pad_bottom; y++) {
        memcpy(&padded->data[(new_height - pad_bottom + y) * new_width + pad_left],
               &img->data[(img->height - 1) * img->width],
               img->width * sizeof(pixel_t));
    }

    // Pad left and right
    for (int y = 0; y < new_height; y++) {
        // Left padding
        for (int x = 0; x < pad_left; x++) {
            padded->data[y * new_width + x] = padded->data[y * new_width + pad_left];
        }
        // Right padding
        for (int x = 0; x < pad_right; x++) {
            padded->data[y * new_width + new_width - pad_right + x] = 
                padded->data[y * new_width + new_width - pad_right - 1];
        }
    }

    return padded;
}

Image* downsample_image(const Image* img, int factor, FilterMode filter_mode, float gaussian_sigma) {
    if (factor <= 1) {
        Image* copy = create_image(img->height, img->width);
        if (copy) {
            memcpy(copy->data, img->data, sizeof(pixel_t) * img->height * img->width);
        }
        return copy;
    }

    // Apply pre-filtering if using Gaussian
    Image* filtered_img = NULL;
    if (filter_mode == FILTER_GAUSSIAN) {
        filtered_img = apply_gaussian_filter(img, gaussian_sigma);
        if (!filtered_img) return NULL;
    } else {
        filtered_img = (Image*)img;  // Use original for box filter
    }

    int new_height = img->height / factor;
    int new_width = img->width / factor;
    Image* downsampled = create_image(new_height, new_width);
    if (!downsampled) {
        if (filter_mode == FILTER_GAUSSIAN) free_image(filtered_img);
        return NULL;
    }

    // Box filter downsampling
    float scale = 1.0f / (factor * factor);
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < factor; ky++) {
                for (int kx = 0; kx < factor; kx++) {
                    int orig_y = y * factor + ky;
                    int orig_x = x * factor + kx;
                    sum += filtered_img->data[orig_y * filtered_img->width + orig_x];
                }
            }
            downsampled->data[y * new_width + x] = sum * scale;
        }
    }

    if (filter_mode == FILTER_GAUSSIAN) free_image(filtered_img);
    return downsampled;
}

// Distance metric functions
float compute_patch_distance_l1(const Image* ref_img, const Image* target_img,
                              int ref_x, int ref_y, int target_x, int target_y,
                              int patch_size) {
    float dist = 0.0f;
    
    for (int y = 0; y < patch_size; y++) {
        for (int x = 0; x < patch_size; x++) {
            int rx = ref_x + x;
            int ry = ref_y + y;
            int tx = target_x + x;
            int ty = target_y + y;
            
            if (rx >= ref_img->width || ry >= ref_img->height ||
                tx >= target_img->width || ty >= target_img->height) {
                return FLT_MAX;
            }
            
            float diff = ref_img->data[ry * ref_img->width + rx] - 
                        target_img->data[ty * target_img->width + tx];
            dist += fabsf(diff);
        }
    }
    
    return dist;
}

float compute_patch_distance_l2(const Image* ref_img, const Image* target_img,
                              int ref_x, int ref_y, int target_x, int target_y,
                              int patch_size) {
    float dist = 0.0f;
    
    for (int y = 0; y < patch_size; y++) {
        for (int x = 0; x < patch_size; x++) {
            int rx = ref_x + x;
            int ry = ref_y + y;
            int tx = target_x + x;
            int ty = target_y + y;
            
            if (rx >= ref_img->width || ry >= ref_img->height ||
                tx >= target_img->width || ty >= target_img->height) {
                return FLT_MAX;
            }
            
            float diff = ref_img->data[ry * ref_img->width + rx] - 
                        target_img->data[ty * target_img->width + tx];
            dist += diff * diff;
        }
    }
    
    return dist;
}

// Core block matching functions
ImagePyramid* init_block_matching(const Image* ref_img, const BlockMatchingParams* params) {
    ImagePyramid* pyramid = create_image_pyramid(params->num_levels);
    if (!pyramid) return NULL;

    // Copy factors
    memcpy(pyramid->factors, params->factors, sizeof(int) * params->num_levels);

    // Create first level (possibly with padding)
    Image* padded = pad_image(ref_img, params->pad_top, params->pad_bottom,
                            params->pad_left, params->pad_right);
    if (!padded) {
        free_image_pyramid(pyramid);
        return NULL;
    }

    pyramid->levels[0] = padded;

    // Create subsequent levels with specified filter mode
    for (int i = 1; i < params->num_levels; i++) {
        pyramid->levels[i] = downsample_image(pyramid->levels[i-1], 
                                            params->factors[i],
                                            params->filter_mode,
                                            params->gaussian_sigma);
        if (!pyramid->levels[i]) {
            free_image_pyramid(pyramid);
            return NULL;
        }
    }

    return pyramid;
}

static void search_best_match(const Image* ref_img, const Image* target_img,
                            int patch_y, int patch_x, int patch_size,
                            int search_radius, bool use_l1,
                            float current_x, float current_y,
                            float* best_x, float* best_y) {
    float min_dist = FLT_MAX;
    int ref_patch_x = patch_x * patch_size;
    int ref_patch_y = patch_y * patch_size;

    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            int target_x = ref_patch_x + (int)current_x + dx;
            int target_y = ref_patch_y + (int)current_y + dy;
            
            float dist;
            if (use_l1) {
                dist = compute_patch_distance_l1(ref_img, target_img,
                                              ref_patch_x, ref_patch_y,
                                              target_x, target_y,
                                              patch_size);
            } else {
                dist = compute_patch_distance_l2(ref_img, target_img,
                                              ref_patch_x, ref_patch_y,
                                              target_x, target_y,
                                              patch_size);
            }

            if (dist < min_dist) {
                min_dist = dist;
                *best_x = current_x + dx;
                *best_y = current_y + dy;
            }
        }
    }
}

AlignmentMap* align_image_block_matching(const Image* img, const ImagePyramid* reference_pyramid,
                                       const BlockMatchingParams* params) {
    printf("Starting alignment with image dimensions: %dx%d\n", img->width, img->height);
    if (!img || !reference_pyramid || !params) {
        fprintf(stderr, "Null pointer passed to align_image_block_matching\n");
        return NULL;
    }

    printf("Number of pyramid levels: %d\n", reference_pyramid->num_levels);
    
    // Create pyramid for target image with same parameters
    ImagePyramid* target_pyramid = init_block_matching(img, params);
    if (!target_pyramid) {
        fprintf(stderr, "Failed to create target pyramid\n");
        return NULL;
    }
    
    AlignmentMap* current_alignment = NULL;
    
    // Process from coarsest to finest level
    for (int level = params->num_levels - 1; level >= 0; level--) {
        printf("Processing pyramid level %d\n", level);
        
        // Add validation for tile_sizes array
        if (!params->tile_sizes) {
            fprintf(stderr, "tile_sizes array is NULL\n");
            free_image_pyramid(target_pyramid);
            return NULL;
        }
        
        int tile_size = params->tile_sizes[level];
        printf("Tile size: %d\n", tile_size);
        
        if (!reference_pyramid->levels[level]) {
            fprintf(stderr, "Null reference pyramid level %d\n", level);
            free_image_pyramid(target_pyramid);
            return NULL;
        }

        if (!target_pyramid->levels[level]) {
            fprintf(stderr, "Null target pyramid level %d\n", level);
            free_image_pyramid(target_pyramid);
            return NULL;
        }
        
        // Add validation for image dimensions and tile size
        /*
        if (tile_size <= 0 || 
            reference_pyramid->levels[level]->width % tile_size != 0 ||
            reference_pyramid->levels[level]->height % tile_size != 0) 
            {
                fprintf(stderr, "Invalid tile size %d for level %d (dimensions: %dx%d)\n",
                        tile_size, level,
                        reference_pyramid->levels[level]->width,
                        reference_pyramid->levels[level]->height);
                free_image_pyramid(target_pyramid);
                return NULL;
            }
        */

                fprintf(stderr, "Invalid tile size %d for level %d (dimensions: %dx%d)\n",
                        tile_size, level,
                        reference_pyramid->levels[level]->width,
                        reference_pyramid->levels[level]->height);


        int n_patches_y = (reference_pyramid->levels[level]->height + tile_size - 1) / tile_size;
        int n_patches_x = (reference_pyramid->levels[level]->width + tile_size - 1) / tile_size;

        
        printf("Number of patches: %dx%d\n", n_patches_x, n_patches_y);

        // Create or upscale alignment map
        AlignmentMap* level_alignment = create_alignment_map(n_patches_y, n_patches_x, tile_size);
        if (!level_alignment) {
            fprintf(stderr, "Failed to create level alignment map\n");
            if (current_alignment) free_alignment_map(current_alignment);
            free_image_pyramid(target_pyramid);
            return NULL;
        }

        if (current_alignment) {
            printf("Upscaling and refining previous alignment\n");
            float scale = (float)params->factors[level];
            int prev_width = current_alignment->width;
            int prev_height = current_alignment->height;
            
            // First do basic upscaling
            for (int y = 0; y < level_alignment->height; y++) {
                for (int x = 0; x < level_alignment->width; x++) {
                    int prev_y = y / scale;
                    int prev_x = x / scale;
                    
                    // Test three candidates and get best alignment
                    float refined_x, refined_y;
                    refine_alignment_with_candidates(
                        reference_pyramid->levels[level],
                        target_pyramid->levels[level],
                        y, x, tile_size,
                        current_alignment,
                        scale, prev_width, prev_height,
                        &refined_x, &refined_y,
                        params
                    );
                    
                    level_alignment->data[y * level_alignment->width + x].x = refined_x;
                    level_alignment->data[y * level_alignment->width + x].y = refined_y;
                }
            }
            
            free_alignment_map(current_alignment);
        }

        printf("Starting block matching for level %d\n", level);
        // Perform block matching at current level
        for (int py = 0; py < n_patches_y; py++) {
            for (int px = 0; px < n_patches_x; px++) {
                float current_x = level_alignment->data[py * level_alignment->width + px].x;
                float current_y = level_alignment->data[py * level_alignment->width + px].y;
                printf("current x = %f; current y = %f\n", current_x, current_y);

                float best_x = current_x;
                float best_y = current_y;
                
                // Search in neighborhood
                float min_dist = FLT_MAX;
                for (int dy = -params->search_radii[level]; dy <= params->search_radii[level]; dy++) {
                    for (int dx = -params->search_radii[level]; dx <= params->search_radii[level]; dx++) {
                        // Add boundary checks
                        int target_x = px * tile_size + dx;
                        int target_y = py * tile_size + dy;
                        
                        if (target_x < 0 || target_x + tile_size > target_pyramid->levels[level]->width ||
                            target_y < 0 || target_y + tile_size > target_pyramid->levels[level]->height) {
                            continue;  // Skip invalid positions
                        }

                        float dist;
                        if (params->use_l1_dist[level]) {
                            dist = compute_patch_distance_l1(reference_pyramid->levels[level],
                                                          target_pyramid->levels[level],
                                                          px * tile_size, py * tile_size,
                                                          target_x, target_y,
                                                          tile_size);
                        } else {
                            dist = compute_patch_distance_l2(reference_pyramid->levels[level],
                                                          target_pyramid->levels[level],
                                                          px * tile_size, py * tile_size,
                                                          target_x, target_y,
                                                          tile_size);
                        }

                        if (dist < min_dist) {
                            min_dist = dist;
                            best_x = dx;  // Store relative displacement
                            best_y = dy;
                        }
                    }
                }
                
                // Update alignment with relative displacement
                level_alignment->data[py * level_alignment->width + px].x = best_x;
                level_alignment->data[py * level_alignment->width + px].y = best_y;

                if (params->debug_mode) {
                    printf("Patch (%d,%d): Best alignment dx=%.2f dy=%.2f\n", 
                           py, px, best_x, best_y);
                }
            }
        }

        current_alignment = level_alignment;
    }

    free_image_pyramid(target_pyramid);
    return current_alignment;
} 

// Add these helper functions for Gaussian filtering
static void compute_gaussian_kernel_1d(float* kernel, int radius, float sigma) {
    int size = 2 * radius + 1;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float x = (float)(i - radius);
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

static Image* apply_gaussian_filter(const Image* img, float sigma) {
    int radius = (int)(4.0f * sigma + 0.5f);  // Same as Python implementation
    int kernel_size = 2 * radius + 1;
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    if (!kernel) return NULL;
    
    compute_gaussian_kernel_1d(kernel, radius, sigma);
    
    // Create temporary images for separable convolution
    Image* temp = create_image(img->height, img->width);
    Image* result = create_image(img->height, img->width);
    if (!temp || !result) {
        free(kernel);
        free_image(temp);
        free_image(result);
        return NULL;
    }
    
    // Horizontal pass
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int k = -radius; k <= radius; k++) {
                int xk = x + k;
                if (xk >= 0 && xk < img->width) {
                    float weight = kernel[k + radius];
                    sum += img->data[y * img->width + xk] * weight;
                    weight_sum += weight;
                }
            }
            
            temp->data[y * img->width + x] = sum / weight_sum;
        }
    }
    
    // Vertical pass
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int k = -radius; k <= radius; k++) {
                int yk = y + k;
                if (yk >= 0 && yk < img->height) {
                    float weight = kernel[k + radius];
                    sum += temp->data[yk * img->width + x] * weight;
                    weight_sum += weight;
                }
            }
            
            result->data[y * img->width + x] = sum / weight_sum;
        }
    }
    
    free(kernel);
    free_image(temp);
    return result;
}

static void refine_alignment_with_candidates(const Image* ref_img, const Image* target_img,
                                           int patch_y, int patch_x, int patch_size,
                                           const AlignmentMap* prev_alignment,
                                           float scale_factor, int prev_width, int prev_height,
                                           float* best_x, float* best_y,
                                           const BlockMatchingParams* params) {
    // Input validation
    if (!ref_img || !target_img || !prev_alignment || !best_x || !best_y) {
        fprintf(stderr, "Null pointer passed to refine_alignment_with_candidates\n");
        return;
    }

    if (patch_size <= 0 || scale_factor <= 0) {
        fprintf(stderr, "Invalid parameters: patch_size=%d, scale_factor=%f\n", 
                patch_size, scale_factor);
        return;
    }

    // Create debug info if needed
    RefinementDebugInfo debug_info = {0};
    if (params->save_refinement_debug) {
        debug_info.patch_x = patch_x;
        debug_info.patch_y = patch_y;
    }
    
    // Get current patch position
    int ref_patch_x = patch_x * patch_size;
    int ref_patch_y = patch_y * patch_size;
    
    // Check patch bounds
    if (ref_patch_x + patch_size > ref_img->width || 
        ref_patch_y + patch_size > ref_img->height) {
        fprintf(stderr, "Patch exceeds image bounds at (%d,%d)\n", patch_x, patch_y);
        return;
    }
    
    // Previous level indices
    int prev_x = patch_x / scale_factor;
    int prev_y = patch_y / scale_factor;
    
    if (prev_x >= prev_width || prev_y >= prev_height) {
        fprintf(stderr, "Invalid previous level indices: (%d,%d)\n", prev_x, prev_y);
        return;
    }
    
    // Initialize with current alignment
    float min_dist = FLT_MAX;
    *best_x = prev_alignment->data[prev_y * prev_width + prev_x].x * scale_factor;
    *best_y = prev_alignment->data[prev_y * prev_width + prev_x].y * scale_factor;
    
    if (params->save_refinement_debug) {
        debug_info.original_x = *best_x;
        debug_info.original_y = *best_y;
    }
    
    // Test three candidates:
    // 1. Current alignment
    float dist = compute_patch_distance_l1(ref_img, target_img,
                                         ref_patch_x, ref_patch_y,
                                         ref_patch_x + (int)*best_x,
                                         ref_patch_y + (int)*best_y,
                                         patch_size);
    min_dist = dist;
    
    if (params->save_refinement_debug) {
        debug_info.distances[0] = dist;
    }
    
    // 2. Vertical shift candidate
    int vert_y_idx = prev_y + 1 < prev_height ? prev_y + 1 : prev_y - 1;
    float vert_x_flow = prev_alignment->data[vert_y_idx * prev_width + prev_x].x * scale_factor;
    float vert_y_flow = prev_alignment->data[vert_y_idx * prev_width + prev_x].y * scale_factor;
    
    if (params->save_refinement_debug) {
        debug_info.vertical_x = vert_x_flow;
        debug_info.vertical_y = vert_y_flow;
    }
    
    dist = compute_patch_distance_l1(ref_img, target_img,
                                   ref_patch_x, ref_patch_y,
                                   ref_patch_x + (int)vert_x_flow,
                                   ref_patch_y + (int)vert_y_flow,
                                   patch_size);
    if (params->save_refinement_debug) {
        debug_info.distances[1] = dist;
    }
    
    if (dist < min_dist) {
        min_dist = dist;
        *best_x = vert_x_flow;
        *best_y = vert_y_flow;
    }
    
    // 3. Horizontal shift candidate
    int horiz_x_idx = prev_x + 1 < prev_width ? prev_x + 1 : prev_x - 1;
    float horiz_x_flow = prev_alignment->data[prev_y * prev_width + horiz_x_idx].x * scale_factor;
    float horiz_y_flow = prev_alignment->data[prev_y * prev_width + horiz_x_idx].y * scale_factor;
    
    if (params->save_refinement_debug) {
        debug_info.horizontal_x = horiz_x_flow;
        debug_info.horizontal_y = horiz_y_flow;
    }
    
    dist = compute_patch_distance_l1(ref_img, target_img,
                                   ref_patch_x, ref_patch_y,
                                   ref_patch_x + (int)horiz_x_flow,
                                   ref_patch_y + (int)horiz_y_flow,
                                   patch_size);
    if (params->save_refinement_debug) {
        debug_info.distances[2] = dist;
    }
    
    if (dist < min_dist) {
        *best_x = horiz_x_flow;
        *best_y = horiz_y_flow;
    }

    // Save debug visualization if enabled
    if (params->save_refinement_debug) {
        if (!params->debug_output_dir) {
            fprintf(stderr, "Debug output directory not set\n");
        } else {
            debug_info.final_x = *best_x;
            debug_info.final_y = *best_y;
            save_refinement_visualization(&debug_info, params->debug_output_dir);
        }
    }
}

// Add visualization function
static void save_refinement_visualization(const RefinementDebugInfo* debug_info, const char* output_dir) {
    if (!debug_info || !output_dir) {
        fprintf(stderr, "Invalid parameters passed to save_refinement_visualization\n");
        return;
    }

    char filename[256];
    snprintf(filename, sizeof(filename), "%s/refinement_patch_%d_%d.txt", 
             output_dir, debug_info->patch_y, debug_info->patch_x);
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open debug file: %s\n", filename);
        return;
    }
    
    fprintf(fp, "Refinement Debug Info for Patch (%d,%d)\n", 
            debug_info->patch_y, debug_info->patch_x);
    fprintf(fp, "Original alignment: (%.2f, %.2f) - Distance: %.2f\n",
            debug_info->original_x, debug_info->original_y, debug_info->distances[0]);
    fprintf(fp, "Vertical candidate: (%.2f, %.2f) - Distance: %.2f\n",
            debug_info->vertical_x, debug_info->vertical_y, debug_info->distances[1]);
    fprintf(fp, "Horizontal candidate: (%.2f, %.2f) - Distance: %.2f\n",
            debug_info->horizontal_x, debug_info->horizontal_y, debug_info->distances[2]);
    fprintf(fp, "Final alignment: (%.2f, %.2f)\n",
            debug_info->final_x, debug_info->final_y);
    
    fclose(fp);
}
