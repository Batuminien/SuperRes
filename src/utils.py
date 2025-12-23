import numpy as np
import cv2

def rgb2yiq(img_rgb):
    """
    Converts RGB image into YIQ format.
    Input: RGB image between 0-1 and has (H, W, 3) 
    Output: (H, W, 3) YIQ image.
    """
    
    yiq_matrix = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322],
                           [0.211, -0.523, 0.312]])
    
    h, w, c = img_rgb.shape
    img_reshaped = img_rgb.reshape(-1, 3)
    img_yiq = np.dot(img_reshaped, yiq_matrix.T)
    return img_yiq.reshape(h, w, c)

def yiq2rgb(img_yiq):
    """
    Converts YIQ image into RGB format.
    Input: YIQ image.
    Output: RGB image (between 0-1)
    """
    
    rgb_matrix = np.array([[1.0, 0.956, 0.621],
                           [1.0, -0.272, -0.647],
                           [1.0, -1.106, 1.703]])
    
    h, w, c = img_yiq.shape
    img_reshaped = img_yiq.reshape(-1, 3)
    img_rgb = np.dot(img_reshaped, rgb_matrix.T)
    img_rgb = np.clip(img_rgb, 0.0, 1.0)
    return img_rgb.reshape(h, w, c)

def imread_normalized(path):
    """
    Reads the image, turns into RGB and normalizes into between 0-1
    """
    
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def imsave_normalized(path, img):
    """
    It arranges 0-255 the input image that has range 0-1 and save it.
    """
    
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)
    
import cv2
import numpy as np

def degrade_gaussian_subsample(hr_img, scale=2, sigma=None):
    """
    Glasner tarzı degrade:
      L = (H * G_sigma) ↓ scale

    Parametreler:
      hr_img : 2D array (tek kanal, 0-1 float)
      scale  : downscale faktörü (örn. 2)
      sigma  : Gaussian std. dev. (None ise teorik s^2-1'dan türetiriz)

    Çıktı:
      lr_img : 2D array, blur + subsample edilmiş
    """

    hr = hr_img.astype(np.float32)
    h, w = hr.shape

    if sigma is None:
        # Glasner tarzı: s^2 - 1'dan türetilmiş sigma
        sigma = np.sqrt(scale ** 2 - 1.0)

    # 1) Gaussian blur
    if sigma > 1e-6:
        blurred = cv2.GaussianBlur(hr, (0, 0),
                                   sigmaX=sigma, sigmaY=sigma)
    else:
        blurred = hr

    # 2) Subsample (decimation)
    lr = blurred[::scale, ::scale]

    return lr
