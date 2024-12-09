#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "linalg.h"

// Add debug logging macro
#ifdef DEBUG_LINALG
#define LOG_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...)
#endif

// Add error checking for matrix operations
static bool is_valid_matrix(const Matrix2x2* M) {
    if (!M) return false;
    // Check for NaN values
    if (isnan(M->a11) || isnan(M->a12) || 
        isnan(M->a21) || isnan(M->a22)) {
        return false;
    }
    return true;
}

// Compute determinant of 2x2 matrix
float matrix2x2_determinant(const Matrix2x2* A) {
    if (!is_valid_matrix(A)) {
        LOG_DEBUG("Invalid matrix in determinant computation");
        return 0.0f;
    }
    return A->a11 * A->a22 - A->a12 * A->a21;
}

// Solve 2x2 linear system Ax = b
void solve_2x2(const Matrix2x2* A, const Vector2D* b, Vector2D* x) {
    float det = matrix2x2_determinant(A);
    if (fabsf(det) < 1e-10f) {
        // Handle singular matrix - return zero vector
        x->x = 0.0f;
        x->y = 0.0f;
        return;
    }

    // Cramer's rule
    x->x = (b->x * A->a22 - b->y * A->a12) / det;
    x->y = (A->a11 * b->y - A->a21 * b->x) / det;
}

// Compute inverse of 2x2 matrix
void matrix2x2_inverse(const Matrix2x2* A, Matrix2x2* inv) {
    float det = matrix2x2_determinant(A);
    if (fabsf(det) < 1e-10f) {
        // Handle singular matrix - return identity
        inv->a11 = 1.0f; inv->a12 = 0.0f;
        inv->a21 = 0.0f; inv->a22 = 1.0f;
        return;
    }

    float inv_det = 1.0f / det;
    inv->a11 = A->a22 * inv_det;
    inv->a12 = -A->a12 * inv_det;
    inv->a21 = -A->a21 * inv_det;
    inv->a22 = A->a11 * inv_det;
}

// Bilinear interpolation for single-channel image
float bilinear_interpolate(const float* image, int height, int width, float y, float x) {
    // Get integer and fractional parts
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float fx = x - x0;
    float fy = y - y0;

    // Clamp coordinates
    x0 = (x0 < 0) ? 0 : ((x0 >= width) ? width - 1 : x0);
    x1 = (x1 < 0) ? 0 : ((x1 >= width) ? width - 1 : x1);
    y0 = (y0 < 0) ? 0 : ((y0 >= height) ? height - 1 : y0);
    y1 = (y1 < 0) ? 0 : ((y1 >= height) ? height - 1 : y1);

    // Get pixel values
    float p00 = image[y0 * width + x0];
    float p01 = image[y0 * width + x1];
    float p10 = image[y1 * width + x0];
    float p11 = image[y1 * width + x1];

    // Interpolate
    float a = p00 * (1.0f - fx) + p01 * fx;
    float b = p10 * (1.0f - fx) + p11 * fx;
    return a * (1.0f - fy) + b * fy;
}

// Bilinear interpolation for RGB image
void bilinear_interpolate_rgb(const Image* image, float y, float x, float* rgb) {
    for (int c = 0; c < image->channels; c++) {
        rgb[c] = bilinear_interpolate(image->data + c * image->height * image->width,
                                    image->height, image->width, y, x);
    }
}

// Compute x gradient using central differences
void compute_gradient_x(const float* image, float* dx, int height, int width) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float left = (x > 0) ? image[y * width + (x-1)] : image[y * width + x];
            float right = (x < width-1) ? image[y * width + (x+1)] : image[y * width + x];
            dx[y * width + x] = 0.5f * (right - left);
        }
    }
}

// Compute y gradient using central differences
void compute_gradient_y(const float* image, float* dy, int height, int width) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float top = (y > 0) ? image[(y-1) * width + x] : image[y * width + x];
            float bottom = (y < height-1) ? image[(y+1) * width + x] : image[y * width + x];
            dy[y * width + x] = 0.5f * (bottom - top);
        }
    }
}

// Compute both gradients at once
void compute_gradients(const Image* image, float* dx, float* dy) {
    for (int c = 0; c < image->channels; c++) {
        const float* channel = image->data + c * image->height * image->width;
        float* dx_channel = dx + c * image->height * image->width;
        float* dy_channel = dy + c * image->height * image->width;
        
        compute_gradient_x(channel, dx_channel, image->height, image->width);
        compute_gradient_y(channel, dy_channel, image->height, image->width);
    }
}

// Create 1D Gaussian kernel
float gaussian_kernel_1d(float x, float sigma) {
    const float inv_sqrt_2pi = 0.3989422804f; // 1/sqrt(2*pi)
    return inv_sqrt_2pi / sigma * expf(-0.5f * x * x / (sigma * sigma));
}

// Create Gaussian kernel for given size and sigma
void create_gaussian_kernel(float* kernel, int size, float sigma) {
    int radius = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float x = (float)(i - radius);
        kernel[i] = gaussian_kernel_1d(x, sigma);
        sum += kernel[i];
    }

    // Normalize kernel
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        kernel[i] *= inv_sum;
    }
}

// 2D convolution with separable kernel
void convolve_2d(const float* input, float* output, const float* kernel,
                 int height, int width, int kernel_size) {
    float* temp = (float*)malloc(height * width * sizeof(float));
    if (!temp) return;

    int radius = kernel_size / 2;

    // Horizontal pass
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int xk = x + k;
                xk = (xk < 0) ? 0 : ((xk >= width) ? width - 1 : xk);
                sum += input[y * width + xk] * kernel[k + radius];
            }
            temp[y * width + x] = sum;
        }
    }

    // Vertical pass
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int yk = y + k;
                yk = (yk < 0) ? 0 : ((yk >= height) ? height - 1 : yk);
                sum += temp[yk * width + x] * kernel[k + radius];
            }
            output[y * width + x] = sum;
        }
    }

    free(temp);
}

// Apply Gaussian blur to image
void gaussian_blur(float* image, int height, int width, float sigma) {
    if (sigma < 1e-6f) return;  // Skip if sigma is too small

    // Create Gaussian kernel
    int kernel_size = (int)(6.0f * sigma);  // 3 sigma on each side
    kernel_size = (kernel_size % 2 == 0) ? kernel_size + 1 : kernel_size;  // Ensure odd size
    
    float* kernel = (float*)malloc(kernel_size * sizeof(float));
    if (!kernel) return;

    create_gaussian_kernel(kernel, kernel_size, sigma);
    convolve_2d(image, image, kernel, height, width, kernel_size);

    free(kernel);
} 

float quad_mat_prod(const Matrix2x2* A, float x1, float x2) {
    if (!is_valid_matrix(A)) {
        LOG_DEBUG("Invalid matrix in quad_mat_prod");
        return 0.0f;
    }
    
    float result = A->a11 * x1 * x1 + 
                  (A->a12 + A->a21) * x1 * x2 + 
                  A->a22 * x2 * x2;
    
    LOG_DEBUG("quad_mat_prod: [%.3f %.3f; %.3f %.3f] * [%.3f; %.3f] = %.3f",
              A->a11, A->a12, A->a21, A->a22, x1, x2, result);
    return result;
}

void get_eigen_val_2x2(const Matrix2x2* M, float* eigenvals) {
    if (!is_valid_matrix(M) || !eigenvals) {
        LOG_DEBUG("Invalid input to get_eigen_val_2x2");
        if (eigenvals) {
            eigenvals[0] = eigenvals[1] = 0.0f;
        }
        return;
    }

    float a = 1.0f;
    float b = -(M->a11 + M->a22);
    float c = M->a11 * M->a22 - M->a12 * M->a21;
    
    LOG_DEBUG("Characteristic equation: x^2 + %.3fx + %.3f = 0", b, c);
    get_real_polyroots_2(a, b, c, eigenvals);
    
    LOG_DEBUG("Eigenvalues: %.3f, %.3f", eigenvals[0], eigenvals[1]);
}

void get_real_polyroots_2(float a, float b, float c, float* roots) {
    // Ensure delta is not negative due to numerical instability
    float delta = fmaxf(b*b - 4*a*c, 0.0f);
    float sqrt_delta = sqrtf(delta);
    
    float r1 = (-b + sqrt_delta)/(2*a);
    float r2 = (-b - sqrt_delta)/(2*a);
    
    // Sort by magnitude
    if (fabsf(r1) >= fabsf(r2)) {
        roots[0] = r1;
        roots[1] = r2;
    } else {
        roots[0] = r2;
        roots[1] = r1;
    }
}

void get_eigen_vect_2x2(const Matrix2x2* M, const float* eigenvals, 
                        Vector2D* e1, Vector2D* e2) {
    if (!is_valid_matrix(M) || !eigenvals || !e1 || !e2) {
        LOG_DEBUG("Invalid input to get_eigen_vect_2x2");
        return;
    }

    LOG_DEBUG("Computing eigenvectors for matrix [%.3f %.3f; %.3f %.3f]",
              M->a11, M->a12, M->a21, M->a22);

    if (M->a12 == 0.0f && M->a11 == M->a22) {
        LOG_DEBUG("Matrix is multiple of identity");
        e1->x = 1.0f; e1->y = 0.0f;
        e2->x = 0.0f; e2->y = 1.0f;
    } else {
        // Compute first eigenvector
        e1->x = M->a11 + M->a12 - eigenvals[1];
        e1->y = M->a21 + M->a22 - eigenvals[1];
        
        if (e1->x == 0.0f) {
            e1->y = 1.0f;
            e2->x = 1.0f;
            e2->y = 0.0f;
        } else if (e1->y == 0.0f) {
            e1->x = 1.0f;
            e2->x = 0.0f;
            e2->y = 1.0f;
        } else {
            // Normalize first eigenvector
            float norm = sqrtf(e1->x * e1->x + e1->y * e1->y);
            e1->x /= norm;
            e1->y /= norm;
            
            // Compute second eigenvector (orthogonal to first)
            float sign = (e1->x >= 0.0f) ? 1.0f : -1.0f;
            e2->y = fabsf(e1->x);
            e2->x = -e1->y * sign;
        }
    }

    LOG_DEBUG("Eigenvector 1: [%.3f; %.3f]", e1->x, e1->y);
    LOG_DEBUG("Eigenvector 2: [%.3f; %.3f]", e2->x, e2->y);
}

void get_eigen_elmts_2x2(const Matrix2x2* M, float* eigenvals, 
                        Vector2D* e1, Vector2D* e2) {
    get_eigen_val_2x2(M, eigenvals);
    get_eigen_vect_2x2(M, eigenvals, e1, e2);
}

void interpolate_cov(const Matrix2x2 covs[2][2], const Vector2D* center_pos, 
                    Matrix2x2* interpolated_cov) {
    if (!covs || !center_pos || !interpolated_cov) {
        LOG_DEBUG("Invalid input to interpolate_cov");
        return;
    }

    // Validate input matrices
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (!is_valid_matrix(&covs[i][j])) {
                LOG_DEBUG("Invalid covariance matrix at position [%d][%d]", i, j);
                return;
            }
        }
    }

    float pos_x = center_pos->x - floorf(center_pos->x);
    float pos_y = center_pos->y - floorf(center_pos->y);
    
    LOG_DEBUG("Interpolating at position (%.3f, %.3f)", pos_x, pos_y);

    // Compute weights
    float w00 = (1-pos_x) * (1-pos_y);
    float w01 = pos_x * (1-pos_y);
    float w10 = (1-pos_x) * pos_y;
    float w11 = pos_x * pos_y;

    LOG_DEBUG("Interpolation weights: %.3f, %.3f, %.3f, %.3f", 
              w00, w01, w10, w11);

    // For a11
    interpolated_cov->a11 = 
        covs[0][0].a11 * w00 +
        covs[0][1].a11 * w01 +
        covs[1][0].a11 * w10 +
        covs[1][1].a11 * w11;
    
    // For a12
    interpolated_cov->a12 = 
        covs[0][0].a12 * w00 +
        covs[0][1].a12 * w01 +
        covs[1][0].a12 * w10 +
        covs[1][1].a12 * w11;
    
    // For a21
    interpolated_cov->a21 = 
        covs[0][0].a21 * w00 +
        covs[0][1].a21 * w01 +
        covs[1][0].a21 * w10 +
        covs[1][1].a21 * w11;
    
    // For a22
    interpolated_cov->a22 = 
        covs[0][0].a22 * w00 +
        covs[0][1].a22 * w01 +
        covs[1][0].a22 * w10 +
        covs[1][1].a22 * w11;

    LOG_DEBUG("Interpolated matrix: [%.3f %.3f; %.3f %.3f]",
              interpolated_cov->a11, interpolated_cov->a12,
              interpolated_cov->a21, interpolated_cov->a22);
}