#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "common.h"
#include "block_matching.h"
#include "ica.h"
#include "utils.h"

// Structure to hold debug information
typedef struct {
    AlignmentMap** flow_history;  // Array of flow maps for each iteration
    int num_flows;               // Number of stored flow maps
} DebugInfo;

// Function to create debug info structure
DebugInfo* create_debug_info(int max_flows) {
    DebugInfo* debug = (DebugInfo*)malloc(sizeof(DebugInfo));
    if (!debug) return NULL;

    debug->flow_history = (AlignmentMap**)calloc(max_flows, sizeof(AlignmentMap*));
    if (!debug->flow_history) {
        free(debug);
        return NULL;
    }
    debug->num_flows = 0;
    return debug;
}

// Function to free debug info
void free_debug_info(DebugInfo* debug) {
    if (debug) {
        for (int i = 0; i < debug->num_flows; i++) {
            free_alignment_map(debug->flow_history[i]);
        }
        free(debug->flow_history);
        free(debug);
    }
}

// Function to add flow to debug history
void add_flow_to_debug(DebugInfo* debug, const AlignmentMap* flow) {
    if (!debug || !flow) return;
    
    AlignmentMap* flow_copy = create_alignment_map(flow->height, flow->width, flow->patch_size);
    if (!flow_copy) return;
    
    memcpy(flow_copy->data, flow->data, sizeof(Alignment) * flow->height * flow->width);
    debug->flow_history[debug->num_flows++] = flow_copy;
}

// Function to save flow visualization
void save_flow_visualization(const char* filename, const AlignmentMap* flow) {
    Image* vis = create_image(flow->height, flow->width * 2);  // 2 channels: x and y flow
    if (!vis) return;

    // Normalize and store flow values
    float min_x = FLT_MAX, max_x = -FLT_MAX;
    float min_y = FLT_MAX, max_y = -FLT_MAX;

    // Find min/max values
    for (int i = 0; i < flow->height * flow->width; i++) {
        min_x = fminf(min_x, flow->data[i].x);
        max_x = fmaxf(max_x, flow->data[i].x);
        min_y = fminf(min_y, flow->data[i].y);
        max_y = fmaxf(max_y, flow->data[i].y);
    }

    // Normalize and store
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    if (range_x < 1e-6f) range_x = 1.0f;
    if (range_y < 1e-6f) range_y = 1.0f;

    for (int i = 0; i < flow->height * flow->width; i++) {
        vis->data[i * 2] = (flow->data[i].x - min_x) / range_x;
        vis->data[i * 2 + 1] = (flow->data[i].y - min_y) / range_y;
    }

    save_image(filename, vis);
    free_image(vis);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <ref_image> <target_image> <output_prefix> [options]\n", argv[0]);
        printf("Options:\n");
        printf("  --debug            Enable debug mode\n");
        printf("  --verbose N        Set verbosity level (0-3)\n");
        printf("  --tile-size N      Set base tile size (default: 16)\n");
        printf("  --ica-iters N      Set ICA iterations (default: 3)\n");
        printf("  --sigma N          Set Gaussian blur sigma (default: 0.0)\n");
        return 1;
    }

    const char* ref_path = argv[1];
    const char* target_path = argv[2];
    const char* output_prefix = argv[3];

    // Parse options
    bool debug_mode = false;
    int verbose = 1;
    int tile_size = 16;
    int ica_iterations = 3;
    float sigma_blur = 0.0f;

    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--debug") == 0) {
            debug_mode = true;
        } else if (strcmp(argv[i], "--verbose") == 0 && i + 1 < argc) {
            verbose = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc) {
            tile_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ica-iters") == 0 && i + 1 < argc) {
            ica_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sigma") == 0 && i + 1 < argc) {
            sigma_blur = atof(argv[++i]);
        }
    }

    // Create debug info if needed
    DebugInfo* debug_info = NULL;
    if (debug_mode) {
        debug_info = create_debug_info(ica_iterations + 1);  // +1 for initial block matching
        if (!debug_info) {
            fprintf(stderr, "Failed to create debug info\n");
            return 1;
        }
    }

    // Load and convert images to grayscale
    if (verbose > 0) printf("Loading images...\n");
    Image* ref_img = load_image(ref_path);
    Image* target_img = load_image(target_path);
    if (!ref_img || !target_img) {
        fprintf(stderr, "Failed to load images\n");
        return 1;
    }

    Image* ref_gray = create_grayscale(ref_img);
    Image* target_gray = create_grayscale(target_img);
    if (!ref_gray || !target_gray) {
        fprintf(stderr, "Failed to convert images to grayscale\n");
        return 1;
    }

    // Initialize block matching parameters
    BlockMatchingParams bm_params = {
        .num_levels = 4,
        .factors = (int[]){1, 2, 4, 4},
        .tile_sizes = (int[]){tile_size, tile_size, tile_size, tile_size/2},
        .search_radii = (int[]){1, 4, 4, 4},
        .use_l1_dist = (bool[]){true, false, false, false},
        .pad_top = 0,
        .pad_bottom = 0,
        .pad_left = 0,
        .pad_right = 0,
        .debug_mode = debug_mode,
        .max_iterations = 10
    };

    // Initialize ICA parameters
    ICAParams ica_params = {
        .sigma_blur = sigma_blur,
        .num_iterations = ica_iterations,
        .tile_size = tile_size
    };

    // Step 1: Block Matching
    if (verbose > 0) printf("Performing block matching...\n");
    ImagePyramid* ref_pyramid = init_block_matching(ref_gray, &bm_params);
    if (!ref_pyramid) {
        fprintf(stderr, "Failed to initialize block matching\n");
        goto cleanup;
    }

    AlignmentMap* initial_flow = align_image_block_matching(target_gray, ref_pyramid, &bm_params);
    if (!initial_flow) {
        fprintf(stderr, "Block matching failed\n");
        goto cleanup;
    }

    if (debug_mode) {
        add_flow_to_debug(debug_info, initial_flow);
        char bm_flow_path[256];
        snprintf(bm_flow_path, sizeof(bm_flow_path), "%s_bm_flow.png", output_prefix);
        save_flow_visualization(bm_flow_path, initial_flow);
    }

    // Step 2: ICA Refinement
    if (verbose > 0) printf("Performing ICA refinement...\n");
    ImageGradients* ref_grads = init_ica(ref_gray, &ica_params);
    if (!ref_grads) {
        fprintf(stderr, "Failed to compute reference gradients\n");
        goto cleanup;
    }

    HessianMatrix* hessian = compute_hessian(ref_grads, ica_params.tile_size);
    if (!hessian) {
        fprintf(stderr, "Failed to compute Hessian\n");
        goto cleanup;
    }

    AlignmentMap* final_flow = refine_alignment_ica(ref_gray, target_gray,
                                                   ref_grads, hessian,
                                                   initial_flow, &ica_params);
    if (!final_flow) {
        fprintf(stderr, "ICA refinement failed\n");
        goto cleanup;
    }

    if (debug_mode) {
        add_flow_to_debug(debug_info, final_flow);
    }

    // Save final flow visualization
    char final_flow_path[256];
    snprintf(final_flow_path, sizeof(final_flow_path), "%s_final_flow.png", output_prefix);
    save_flow_visualization(final_flow_path, final_flow);

    if (verbose > 0) printf("Processing complete.\n");

cleanup:
    // Cleanup
    free_image(ref_img);
    free_image(target_img);
    free_image(ref_gray);
    free_image(target_gray);
    free_image_pyramid(ref_pyramid);
    free_alignment_map(initial_flow);
    free_alignment_map(final_flow);
    free_image_gradients(ref_grads);
    free_hessian_matrix(hessian);
    free_debug_info(debug_info);

    return 0;
} 