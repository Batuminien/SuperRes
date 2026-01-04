import cv2
import numpy as np

def create_gaussian_pyramid(
    image,
    scale_factor=1.25,
    min_size=32,
    max_depth=6
):
    """
    Generates a Gaussian pyramid (Image Cascade) for multi-scale analysis.
    
    This function iteratively blurs and downsamples the input image to create
    a pyramid structure. This is crucial for patch-based SR methods (like Glasner et al.)
    to find similar patches across different scales within the same image.
    
    Args:
        image (np.ndarray): Input 2D grayscale image (Y channel) with shape (H, W).
        scale_factor (float, optional): The downscaling factor between pyramid levels. Paper typically uses 1.25. Defaults to 1.25
        min_size (int, optional): The minimum allowed dimension (height or width) for a pyramid level.
                                  Recursion stops if the image gets smaller than this. Defaults to 32.
        max_depth (int, optional): The maximum number of downscaled levels to generate. Defaults to 6
    
    Returns:
        List[np.ndarray]: A list containing the image pyramid, starting from the original
                          image [Level 0] down to the smallest scale [Level -N]
    """
    
    # Initialize the pyramid with the original high-resolution image (Level 0)
    pyramid = [image.astype(np.float32)]
    current_img = image.astype(np.float32)
    
    for i in range(1, max_depth + 1):
        h, w = current_img.shape
        
        # Calculate new dimensions based on the scale factor
        new_h = int(h / scale_factor)
        new_w = int(w / scale_factor)
        
        # Safety Check: Stop if the image becomes too small
        if new_h < min_size or new_w < min_size:
            break
        
        # Calculate the standart deviation (sigma) for the Gaussian kernel.
        # According to sampling theory, to prevent aliasing during downsampling,
        # the image must be low-pass filtered. The sigma is derived from the scale factor:
        # sigma = sqrt(s^2 - 1)
        sigma = np.sqrt(scale_factor**2 - 1.0)
        
        # Apply Gaussian Blur 
        blurred = cv2.GaussianBlur(current_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        
        # Downsample the image
        # Note: Since we applied a strong Gaussian blur, INTER_NEAREST is acceptable here,
        # though INTER_LINEAR or INTER_AREA is often preferred in standard resize operations.
        current_img = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        pyramid.append(current_img)
        print(f"Pyramid Level -{i}: {current_img.shape}")
        
    return pyramid