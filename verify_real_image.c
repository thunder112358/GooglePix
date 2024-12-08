#include <stdio.h>
#include <stdlib.h>
#include "block_matching.h"
#include "ica.h"

#define YUV420P_U_OFFSET(w,h) ((w)*(h))
#define YUV420P_V_OFFSET(w,h) ((w)*(h) + ((w)*(h))/4)

// Function to load YUV image (assuming YUV420 format)
static Image* load_yuv_image(const char* filename, int width, int height) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open file: %s\n", filename);
        return NULL;
    }
    
    // Create image for Y component only
    Image* img = create_image(height, width);
    if (!img) {
        fprintf(stderr, "Failed to create image\n");
        fclose(fp);
        return NULL;
    }
    
    // Allocate temporary buffer for YUV data
    size_t y_size = width * height;
    size_t uv_size = (width * height) / 4;
    uint8_t* yuv_data = (uint8_t*)malloc(y_size + 2 * uv_size);
    if (!yuv_data) {
        fprintf(stderr, "Failed to allocate YUV buffer\n");
        free_image(img);
        fclose(fp);
        return NULL;
    }
    
    // Read entire YUV420p frame
    size_t total_size = y_size + 2 * uv_size;
    if (fread(yuv_data, 1, total_size, fp) != total_size) {
        fprintf(stderr, "Failed to read YUV data\n");
        free(yuv_data);
        free_image(img);
        fclose(fp);
        return NULL;
    }
    
    // Convert Y plane to float [0,1]
    for (int i = 0; i < y_size; i++) {
        img->data[i] = (float)yuv_data[i] / 255.0f;
    }
    
    // We only use Y component for alignment, so U and V are ignored
    
    free(yuv_data);
    fclose(fp);
    return img;
}

// Function to save visualization as PPM image
static void save_visualization(const char* filename, const Image* ref_img, 
                             const AlignmentMap* alignment, int scale) {
    int width = ref_img->width;
    int height = ref_img->height;
    
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Allocate buffer for RGB image
    uint8_t* rgb = (uint8_t*)malloc(width * height * 3);
    if (!rgb) {
        fclose(fp);
        return;
    }
    
    // Convert grayscale to RGB and draw alignment vectors
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            uint8_t gray = (uint8_t)(ref_img->data[y * width + x] * 255.0f);
            rgb[idx] = gray;     // R
            rgb[idx + 1] = gray; // G
            rgb[idx + 2] = gray; // B
        }
    }
    
    // Draw alignment vectors
    for (int py = 0; py < alignment->height; py++) {
        for (int px = 0; px < alignment->width; px++) {
            int center_x = px * alignment->patch_size + alignment->patch_size / 2;
            int center_y = py * alignment->patch_size + alignment->patch_size / 2;
            float dx = alignment->data[py * alignment->width + px].x * scale;
            float dy = alignment->data[py * alignment->width + px].y * scale;
            
            // Draw arrow (simple line in red)
            int end_x = center_x + (int)dx;
            int end_y = center_y + (int)dy;
            
            // Simple line drawing (Bresenham's algorithm)
            int x0 = center_x, y0 = center_y;
            int x1 = end_x, y1 = end_y;
            int dx_i = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
            int dy_i = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
            int err = (dx_i > dy_i ? dx_i : -dy_i) / 2;
            
            while (1) {
                if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
                    int idx = (y0 * width + x0) * 3;
                    rgb[idx] = 255;   // R (red color for vectors)
                    rgb[idx + 1] = 0; // G
                    rgb[idx + 2] = 0; // B
                }
                if (x0 == x1 && y0 == y1) break;
                int e2 = err;
                if (e2 > -dx_i) { err -= dy_i; x0 += sx; }
                if (e2 < dy_i) { err += dx_i; y0 += sy; }
            }
        }
    }
    
    // Write RGB data
    fwrite(rgb, 1, width * height * 3, fp);
    
    free(rgb);
    fclose(fp);
}

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  Synthetic mode: %s <ref.yuv> <width> <height>\n", argv[0]);
        fprintf(stderr, "  Real image mode: %s <ref.yuv> <target.yuv> <width> <height>\n", argv[0]);
        return 1;
    }
    
    bool synthetic_mode = (argc == 4);
    const char* ref_file;
    const char* target_file = NULL;
    int width, height;
    
    if (synthetic_mode) {
        ref_file = argv[1];
        width = atoi(argv[2]);
        height = atoi(argv[3]);
    } else {
        ref_file = argv[1];
        target_file = argv[2];
        width = atoi(argv[3]);
        height = atoi(argv[4]);
    }
    
    printf("Loading reference image: %s (%dx%d)\n", ref_file, width, height);
    
    // Load reference YUV image
    Image* ref_img = load_yuv_image(ref_file, width, height);
    if (!ref_img) {
        fprintf(stderr, "Failed to load reference image\n");
        return 1;
    }
    printf("Successfully loaded reference image\n");
    
    // Handle target image based on mode
    Image* target_img = NULL;
    if (synthetic_mode) {
        // Create synthetic target image (shifted version)
        target_img = create_image(height, width);
        if (!target_img) {
            fprintf(stderr, "Failed to create synthetic target image\n");
            free_image(ref_img);
            return 1;
        }
        printf("Created synthetic target image\n");
        
        // Create shifted version (simple shift by 5 pixels right, 3 pixels down)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_x = x - 5;
                int src_y = y - 3;
                if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                    target_img->data[y * width + x] = ref_img->data[src_y * width + src_x];
                }
            }
        }
        printf("Created shifted version with dx=5, dy=3\n");
    } else {
        // Load real target image
        printf("Loading target image: %s\n", target_file);
        target_img = load_yuv_image(target_file, width, height);
        if (!target_img) {
            fprintf(stderr, "Failed to load target image\n");
            free_image(ref_img);
            return 1;
        }
        printf("Successfully loaded target image\n");
    }
    
    // Block matching parameters
    BlockMatchingParams bm_params = {
        .num_levels = 4,
        .factors = (int[]){1, 2, 4, 4},
        .tile_sizes = (int[]){16, 16, 16, 8},
        .search_radii = (int[]){1, 4, 4, 4},
        .use_l1_dist = (bool[]){true, false, false, false},
        .pad_top = 0,
        .pad_bottom = 0,
        .pad_left = 0,
        .pad_right = 0,
        .debug_mode = true,
        .filter_mode = FILTER_GAUSSIAN,
        .gaussian_sigma = 0.5f
    };
    printf("Initialized parameters\n");
    
    // Run block matching
    printf("Starting block matching initialization...\n");
    ImagePyramid* ref_pyramid = init_block_matching(ref_img, &bm_params);
    if (!ref_pyramid) {
        fprintf(stderr, "Failed to initialize block matching\n");
        free_image(ref_img);
        free_image(target_img);
        return 1;
    }
    printf("Block matching initialized\n");
    
    printf("Starting image alignment...\n");
    AlignmentMap* alignment = align_image_block_matching(target_img, ref_pyramid, &bm_params);
    if (!alignment) {
        fprintf(stderr, "Failed to align images\n");
        free_image_pyramid(ref_pyramid);
        free_image(ref_img);
        free_image(target_img);
        return 1;
    }
    printf("Image alignment completed\n");
    
    if (alignment) {
        // Save block matching visualization
        save_visualization("block_matching_result.ppm", ref_img, alignment, 5);
        
        // Create and save block matching warped image
        Image* bm_warped = create_warped_image(target_img, alignment);
        if (bm_warped) {
            // Save as PPM
            FILE* fp = fopen("block_matching_warped.ppm", "wb");
            if (fp) {
                fprintf(fp, "P6\n%d %d\n255\n", bm_warped->width, bm_warped->height);
                // Convert float [0,1] to uint8 [0,255]
                uint8_t* rgb = (uint8_t*)malloc(bm_warped->height * bm_warped->width * 3);
                if (rgb) {
                    for (int i = 0; i < bm_warped->height * bm_warped->width; i++) {
                        uint8_t val = (uint8_t)(bm_warped->data[i] * 255.0f);
                        rgb[i*3 + 0] = val;  // R
                        rgb[i*3 + 1] = val;  // G
                        rgb[i*3 + 2] = val;  // B
                    }
                    fwrite(rgb, 1, bm_warped->height * bm_warped->width * 3, fp);
                    free(rgb);
                }
                fclose(fp);
            }
            free_image(bm_warped);
        }
        
        // Add error map visualization for block matching
        ErrorMap* bm_error = create_error_map(ref_img, target_img, alignment);
        if (bm_error) {
            save_error_map_visualization(bm_error, "block_matching_error.ppm");
        }
        
        // Run ICA refinement
        ICAParams ica_params = {
            .sigma_blur = 0.5f,
            .num_iterations = 5,
            .tile_size = 16
        };
        
        ImageGradients* grads = init_ica(ref_img, &ica_params);
        HessianMatrix* hessian = compute_hessian(grads, ica_params.tile_size);
        AlignmentMap* refined_alignment = refine_alignment_ica(ref_img, target_img, grads,
                                                             hessian, alignment, &ica_params);
        
        if (refined_alignment) {
            // Save ICA refinement visualization
            save_visualization("ica_refined_result.ppm", ref_img, refined_alignment, 5);
            
            // Create and save ICA warped image
            Image* ica_warped = create_warped_image(target_img, refined_alignment);
            if (ica_warped) {
                FILE* fp = fopen("ica_refined_warped.ppm", "wb");
                if (fp) {
                    fprintf(fp, "P6\n%d %d\n255\n", ica_warped->width, ica_warped->height);
                    uint8_t* rgb = (uint8_t*)malloc(ica_warped->height * ica_warped->width * 3);
                    if (rgb) {
                        for (int i = 0; i < ica_warped->height * ica_warped->width; i++) {
                            uint8_t val = (uint8_t)(ica_warped->data[i] * 255.0f);
                            rgb[i*3 + 0] = val;
                            rgb[i*3 + 1] = val;
                            rgb[i*3 + 2] = val;
                        }
                        fwrite(rgb, 1, ica_warped->height * ica_warped->width * 3, fp);
                        free(rgb);
                    }
                    fclose(fp);
                }
                free_image(ica_warped);
            }
            
            // Add error map visualization for ICA
            ErrorMap* ica_error = create_error_map(ref_img, target_img, refined_alignment);
            if (ica_error) {
                save_error_map_visualization(ica_error, "ica_refined_error.ppm");
                
                // Compare block matching and ICA errors
                analyze_error_maps(bm_error, ica_error);
                
                // Add region analysis (4x4 grid)
                printf("\nBlock Matching Region Analysis:\n");
                analyze_regions(bm_error, 4, 4);
                
                printf("\nICA Region Analysis:\n");
                analyze_regions(ica_error, 4, 4);
                
                free_error_map(ica_error);
            }
            
            free_alignment_map(refined_alignment);
            free_hessian_matrix(hessian);
            free_image_gradients(grads);
        }
        
        if (bm_error) {
            free_error_map(bm_error);
        }
        
        free_alignment_map(alignment);
    }
    
    free_image_pyramid(ref_pyramid);
    free_image(ref_img);
    free_image(target_img);
    printf("Cleanup completed\n");
    
    return 0;
} 
