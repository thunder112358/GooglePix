#ifndef KERNELS_H
#define KERNELS_H

#include "common.h"

// Kernel types
typedef enum {
    KERNEL_HANDHELD,
    KERNEL_ISO
} KernelType;

// Kernel parameters
typedef struct {
    KernelType type;       // Kernel type (handheld or iso)
    float k_detail;        // Detail preservation parameter
    float k_denoise;       // Denoising parameter
    float d_tr;           // Threshold for detail preservation
    float d_th;           // Threshold for high-frequency content
    float k_shrink;       // Kernel shrinkage parameter
    float k_stretch;      // Kernel stretching parameter
    int window_size;      // Size of the kernel window
} KernelParams;

// Steerable kernel structure
typedef struct {
    float* weights;       // Kernel weights
    float* orientations;  // Kernel orientations
    int size;            // Kernel size
    int count;           // Number of kernels
} SteerableKernels;

// Function declarations
SteerableKernels* estimate_kernels(const Image* image, const KernelParams* params);
void free_steerable_kernels(SteerableKernels* kernels);

// Helper functions
void compute_structure_tensor(const Image* image, float* Ixx, float* Ixy, float* Iyy);
void estimate_local_orientation(const float* Ixx, const float* Ixy, const float* Iyy,
                              float* magnitudes, float* orientations,
                              int height, int width);
void generate_steerable_basis(float* basis, int size, float orientation);
float compute_anisotropy(float lambda1, float lambda2);

// Add to kernels.h
typedef struct {
    float alpha;
    float beta;
} NoiseParams;

// Add GAT parameters
typedef struct {
    float k_detail;
    float k_denoise;
    float d_tr;
    float d_th;
    float k_stretch;
    float k_shrink;
    NoiseParams noise;
} KernelEstimationParams;

// Add function declarations
void apply_gat(float* image, int size, const NoiseParams* noise);
void compute_gradients_prewitt(const Image* image, float* grad_x, float* grad_y);
void estimate_kernel_covariance(const Image* image, 
                              const KernelEstimationParams* params,
                              Matrix2x2* covs, int cov_height, int cov_width);
void compute_gradient_x(const float* image, float* dx, int height, int width);
void compute_gradient_y(const float* image, float* dy, int height, int width);
void gaussian_blur(float* image, int height, int width, float sigma);

// Add visualization function declarations
void save_kernel_visualization(const float* kernel, int size, const char* filename);
void visualize_steerable_kernels(const SteerableKernels* kernels, 
                                const Image* image, 
                                const char* prefix);

// Add to kernels.h
// Structure for kernel response visualization
typedef struct {
    float* response;      // Response map
    int height;
    int width;
    float min_val;
    float max_val;
} KernelResponse;

// Add function declarations
KernelResponse* compute_kernel_response(const Image* image, const float* kernel, int kernel_size);
void save_response_visualization(const KernelResponse* response, const char* filename);
void free_kernel_response(KernelResponse* response);

// Add the clamp function declaration
float clamp(float value, float min, float max);

// Function declarations
SteerableKernels* create_steerable_kernels(int size, int count);
void free_steerable_kernels(SteerableKernels* kernels);

#endif // KERNELS_H 