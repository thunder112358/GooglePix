import numpy as np
from block_matching import init_block_matching, align_image_block_matching

def test_block_matching():
    # Create test images
    height = 64
    width = 64
    
    # Create reference image: diagonal gradient
    ref_img = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            ref_img[y,x] = float(x + y) / (width + height)

    # Create target image: shifted diagonal gradient
    shift_x = 2
    shift_y = 3
    target_img = np.roll(np.roll(ref_img, shift_y, axis=0), shift_x, axis=1)

    # Setup parameters
    options = {
        'verbose': 2
    }
    
    params = {
        'tuning': {
            'factors': [1, 2, 4],
            'tileSizes': [8, 16, 32],
            'distances': ['L1', 'L1', 'L1'],
            'searchRadia': [4, 4, 4]
        }
    }

    # Initialize pyramid
    ref_pyramid = init_block_matching(ref_img, options, params)
    
    # Perform block matching
    alignments = align_image_block_matching(target_img, ref_pyramid, options, params)

    # Print results
    print("Alignment Results:")
    for y in range(alignments.shape[0]):
        for x in range(alignments.shape[1]):
            print(f"Patch ({y},{x}): dx={alignments[y,x,0]:.1f} dy={alignments[y,x,1]:.1f}")

if __name__ == "__main__":
    test_block_matching() 