import numpy as np
import cv2

def rgb2yiq(img_rgb):
    """
    Converts an RGB image into the YIQ color space.
    
    YIQ is the color space used by the NTSC color TV system.
    - Y (Luminance): Represents brightness/intensity (Super-Resolution is applied here).
    - I (In-phase): Represents the orange-blue range.
    - Q (Quadrature): Represents the purple-green range.
    
    Args:
        img_rgb (np.ndarray): Input RGB image with shape (H, W, 3) and values in range [0, 1].
        
    Returns:
        np.ndarray: The image converted to YIQ space with shape (H, W, 3).
    """
    # Standard NTSC conversion matrix
    yiq_matrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322],
                           [0.211, -0.523, 0.312]])
    
    h, w, c = img_rgb.shape
    
    # Flatten pixels to allow matrix multiplication
    img_reshaped = img_rgb.reshape(-1, 3)
    
    # Apply linear transformation: YIQ = RGB * Matrix^T
    img_yiq = np.dot(img_reshaped, yiq_matrix.T)
    
    return img_yiq.reshape(h, w, c)

def yiq2rgb(img_yiq):
    """
    Converts a YIQ image back into the RGB color space.
    
    Args:
        img_yiq (np.ndarray): Input YIQ image with shape (H, W, 3).
        
    Returns:
        np.ndarray: The reconstructed RGB image with shape (H, W, 3), clipped to range [0, 1].
    """
    
    # Inverse NTSC conversion matrix
    rgb_matrix = np.array([[1.0, 0.956, 0.621],
                           [1.0, -0.272, -0.647],
                           [1.0, -1.106, 1.703]])
    
    h, w, c = img_yiq.shape
    img_reshaped = img_yiq.reshape(-1, 3)
    
    # Apply inverse transformation
    img_rgb = np.dot(img_reshaped, rgb_matrix.T)
    
    # Clip values to valid image range [0, 1] to prevent color artifacts
    img_rgb = np.clip(img_rgb, 0.0, 1.0)
    
    return img_rgb.reshape(h, w, c)

def imread_normalized(path):
    """
    Reads an image from disk, converts BGR to RGB, and normalizes pixel values to [0, 1].
    
    Args:
        path (str): File path to the image.
        
    Returns:
        np.ndarray: Loaded image as float32 array in RGB format.
        
    Raises:
        FileNotFoundError: If the image cannot be read from the path.
    """
    
    # OpenCV reads in BGR format by default
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
    
    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] range for numerical stability in SR algorithms
    return img.astype(np.float32) / 255.0

def imsave_normalized(path, img):
    """
    Saves a normalized float image ([0, 1]) to disk as an 8-bit image ([0, 255]).
    
    Args:
        path (str): Destination file path.
        img (np.ndarray): Input RGB image in range [0, 1].
    """
    
    # Clip values to ensure valid color range and convert to 8-bit integer
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    # Convert RGB -> BGR for OpenCV saving compatibility
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(path, img_bgr)

def degrade_gaussian_subsample(hr_img, scale=2, sigma=None):
    """
    Simulates the image acquisition process to generate a Low-Resolution (LR) image.
    
    The degradation model follows the standard formulation in SR literature (e.g., Glasner et al.):
        LR = (HR * Gaussian_Blur) ↓ scale
        
    This involves two steps:
        1. Anti-aliasing Filter: Blurring with a Gaussian kernel to band-limit the signal.
        2. Decimation: Subsampling pixels by the scale factor.

    Args:
        hr_img (np.ndarray): High-Resolution input image (single channel Y or RGB).
        scale (int, optional): Downscaling factor (e.g., 2, 3, 4). Defaults to 2.
        sigma (float, optional): Standard deviation for the Gaussian kernel. 
                                 If None, it is calculated as sqrt(s^2 - 1) to match theoretical requirements.

    Returns:
        np.ndarray: The degraded Low-Resolution image.
    """

    hr = hr_img.astype(np.float32)
    h, w = hr.shape # Handle both 2D (H,W) and 3D (H,W,C) images

    # If sigma is not provided, derive it from the scale factor.
    # Theoretical justification: To prevent aliasing, the cut-off frequency should match the Nyquist rate.
    # A common approximation in SR is sigma = sqrt(scale^2 - 1).
    if sigma is None:
        sigma = np.sqrt(scale ** 2 - 1.0)

    # Apply gaussian blur
    # If sigma is very small, blurring is negligible, so we skip it to save time.
    if sigma > 1e-6:
        blurred = cv2.GaussianBlur(hr, (0, 0),
                                   sigmaX=sigma, sigmaY=sigma)
    else:
        blurred = hr

    # Subsample
    # We take every 'scale'-th pixel (Strided slicing).
    # Note: This assumes the blur step has sufficiently suppressed high frequencies to avoid aliasing.
    lr = blurred[::scale, ::scale]

    return lr
