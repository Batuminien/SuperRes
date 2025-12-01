import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from src.utils import imread_normalized, rgb2yiq, yiq2rgb
from src.sr_solver import SRSolver

def calculate_metrics(original, reconstructed):
    """
    it calculates PSNR and SSIM values between two images.
    Images compares in the Y (Luminance) channel because SR runs in there.
    """
    
    orig = np.clip(original, 0, 1)
    recon = np.clip(reconstructed, 0, 1)
    
    score_psnr = psnr(orig, recon, data_range=1.0)
    
    score_ssim = ssim(orig, recon, data_range=1.0)
    
    return score_psnr, score_ssim

def main():
    gt_path = "input/Set5/Set5/baby.png"
    
    print(f"Ground Truth image is loading: {gt_path}")
    gt_img = imread_normalized(gt_path)
    
    # dividing the dimensions by 4 makes our job easier (we will reduce and enlarge by x2)
    h, w, c = gt_img.shape
    h = (h // 4) * 4
    w = (w // 4) * 4
    gt_img = gt_img[:h, :w, :]
    
    #take y channel
    gt_yiq = rgb2yiq(gt_img)
    gt_y = gt_yiq[:, :, 0]
    
    #create sentetic low resolution
    scale = 0.5
    lr_h, lr_w = int(h * scale), int(w * scale)
    
    #first little gaussian blur after reducing, it prevents aliasing.
    gt_y_blurred = cv2.GaussianBlur(gt_y, (3, 3), 0.5)
    lr_y = cv2.resize(gt_y_blurred, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    
    print(f"Original Shape: {gt_y.shape}")
    print(f"Low Resolution Shape: {lr_y.shape}")
    print("-"*30)
    
    #bicubic interpolation
    print("Bicubic scaling is doing...")
    bicubic_y = cv2.resize(lr_y, (w, h), interpolation=cv2.INTER_CUBIC)
    
    #our method
    print("SR Method is running...")
    solver = SRSolver(lr_y, scale_factor=1.25)
    our_y = solver.upscale(target_scale=2.0)
    
    #if there is shape mismatch
    our_y = cv2.resize(our_y, (w,h))
    
    b_psnr, b_ssim = calculate_metrics(gt_y, bicubic_y)
    print(f"Bicubic -> PSNR: {b_psnr:.4f} dB | SSIM: {b_ssim:.4f}")
    
    #our scores
    o_psnr, o_ssim = calculate_metrics(gt_y, our_y)
    print(f"SR -> PSNR: {o_psnr:.4f} dB | SSIM: {o_ssim:.4f}")
    
    diff_psnr = o_psnr - b_psnr
    if diff_psnr > 0:
        print(f"\nOur method is much better {diff_psnr:.4f} dB.")
    else:
        print(f"\nSorry bro! i tried.")
        
if __name__ == "__main__":
    main()