import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, size=(256, 256)):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(size)
    return np.array(img, dtype=np.float32) / 255.0

def create_shifted_image(image, dx=5, dy=3):
    """Create a shifted version of the image with some noise"""
    from scipy.ndimage import shift
    shifted = shift(image, (dy, dx), mode='reflect')
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.01, shifted.shape)
    shifted = np.clip(shifted + noise, 0, 1)
    return shifted

def visualize_alignment(ref_img, target_img, alignment_map, title="Alignment Visualization"):
    """Visualize the alignment vectors on the image"""
    plt.figure(figsize=(15, 5))
    
    # Plot reference image
    plt.subplot(131)
    plt.imshow(ref_img, cmap='gray')
    plt.title("Reference Image")
    
    # Plot target image
    plt.subplot(132)
    plt.imshow(target_img, cmap='gray')
    plt.title("Target Image")
    
    # Plot alignment vectors
    plt.subplot(133)
    plt.imshow(ref_img, cmap='gray')
    
    # Draw alignment vectors
    patch_size = alignment_map.patch_size
    for y in range(alignment_map.height):
        for x in range(alignment_map.width):
            center_y = y * patch_size + patch_size // 2
            center_x = x * patch_size + patch_size // 2
            dx = alignment_map.data[y * alignment_map.width + x].x
            dy = alignment_map.data[y * alignment_map.width + x].y
            plt.arrow(center_x, center_y, dx*5, dy*5, 
                     color='r', head_width=2, head_length=2)
    
    plt.title(title)
    plt.show()

def main():
    # Load a real image (you can replace with your own image)
    image_path = "test_image.jpg"  # Replace with your image path
    ref_img = load_and_preprocess_image(image_path)
    
    # Create a shifted version as target image
    target_img = create_shifted_image(ref_img, dx=5, dy=3)
    
    # Save images for C program
    np.save("ref_img.npy", ref_img)
    np.save("target_img.npy", target_img)
    
    # Parameters for block matching
    params = {
        'num_levels': 3,
        'factors': [1, 2, 4],
        'tile_sizes': [16, 32, 64],
        'search_radii': [8, 4, 4],
        'use_l1_dist': [True, True, True],
        'pad_top': 0,
        'pad_bottom': 0,
        'pad_left': 0,
        'pad_right': 0
    }
    
    # Run block matching (you'll need to implement this)
    # alignment_map = run_block_matching(ref_img, target_img, params)
    
    # Run ICA refinement (you'll need to implement this)
    # refined_alignment = run_ica_refinement(ref_img, target_img, alignment_map)
    
    # Visualize results
    # visualize_alignment(ref_img, target_img, alignment_map, "Block Matching Result")
    # visualize_alignment(ref_img, target_img, refined_alignment, "ICA Refined Result")

if __name__ == "__main__":
    main() 