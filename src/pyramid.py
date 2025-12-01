import cv2
import numpy as np

def create_gaussian_pyramid(
    image,
    scale_factor=1.25,
    min_size=32,
    max_depth=6
):
    """
    It creates Image Cascade structure from the paper.
    
    Args:
        image (numpy array): Just Y channel (2D array).
        scale_factor (float): Reducing factor (it is 1.25^s in paper).
        min_size (int): Stop when the image being little than that parameter.
        max_depth (int): Maximum depth (it is recommended as 6 in the paper).
        
    Returns:
        list: Reduced image lists [I_0, I_-1, I_-2, ...]
    """
    
    pyramid = []
    current_img = image.copy()
    pyramid.append(current_img)
    
    #sigma = np.sqrt(scale_factor**2 - 1)
    
    print(f"Pyramid Level 0: {current_img.shape}")
    
    for i in range(1, max_depth + 1):
        h, w = current_img.shape
        
        #calculate new sizes
        new_h = int(h / scale_factor)
        new_w = int(w / scale_factor)
        
        #safety check
        if new_h < min_size or new_w < min_size:
            break
        
        #in paper it is Blur and Sample
        #with cv2.resize INTER_CUBIC it does smoothing and reducing well.
        #if there is need, we can make first gaussian blur after that reduce.
        #blurred = cv2.GaussianBlur(current_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        current_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        pyramid.append(current_img)
        print(f"Pyramid Level -{i}: {current_img.shape}")
        
    return pyramid