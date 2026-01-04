import os
import uuid
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.edge_based_sr import OptimizedEdgeSR
from src.wavelet_ibp_sr import run_wavelet_ibp_sr
from typing import Any, Dict

from src.utils import (
    imread_normalized,
    rgb2yiq,
    yiq2rgb,
    imsave_normalized,
    degrade_gaussian_subsample,
)
from src.sr_solver import SRSolver

def run_classical_example_sr(
    input_path: str,
    use_degradation: bool,
    target_scale: int = 2,
    out_dir: str = "static/results"
) -> Dict[str, Any]:
    """
    Orchestrates the complete Super-Resolution pipeline.
    
    This function handles the end-to-end process:
        1.  Loads the input image.
        2.  Prepares the Low-Resolution (LR) input:
            - If use_degradation=True: Artificially degrades (blur+downsample) the input to create a Ground Truth (GT) / LR pair for benchmarking.
            - If use_degradation=False: Treats the input image directly as the LR source (Real-world scenario).
        3.  Converts images to YIQ color space (SR is performed only on Y channel).
        4.  Runs the SR Solver (Example-Based + Classical Back-Projection).
        5.  Reconstructs the color image (Y_sr + I_bicubic + Q_bicubic).
        6.  Computes quantitative metrics (PSNR/SSIM) if Ground Truth is available.
        7.  Saves all intermediate and final results to disk.

    Args:
        input_path (str): Path to the uploaded input image.
        use_degradation (bool): Flag to determine if artificial degradation is applied.
                                - True: Benchmarking mode.
                                - False: Production mode.
        target_scale (int, optional): The upscaling factor (e.g., 2, 3, 4). Defaults to 2.
        out_dir (str, optional): Directory to save results. Defaults to "static/results".

    Returns:
        Dict[str, Any]: A dictionary containing paths to saved images, metrics, and job metadata 
                        suitable for rendering in the web UI.
    """
    
    # Create a unique ID for this processing job to avoid filename collisions
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(out_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    downscale_factor = int(target_scale)
    
    # Load and normalize input image [0, 1]
    img = imread_normalized(input_path)
    hr_rgb = img
    metrics = None
    
    # Benchmarking mode
    if use_degradation:
        # Convert to YIQ to separate Luminance (Y) from Chrominance (I, Q)
        hr_yiq = rgb2yiq(hr_rgb)
        hr_y = hr_yiq[:, :, 0]
        hr_i = hr_yiq[:, :, 1]
        hr_q = hr_yiq[:, :, 2]
        
        # Degrade Y channel
        lr_y = degrade_gaussian_subsample(hr_y, scale=downscale_factor, sigma=None)
        h_lr, w_lr = lr_y.shape
        
        # Resize I and Q channels using simple Bicubic.
        lr_i = cv2.resize(hr_i, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        lr_q = cv2.resize(hr_q, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        
        # Reconstruct and save the simulated LR image
        lr_yiq = np.dstack((lr_y, lr_i, lr_q))
        lr_rgb = yiq2rgb(lr_yiq)
        
        imsave_normalized(os.path.join(job_dir, "lr_degraded.png"), np.clip(lr_rgb, 0, 1))
    # Real-world mode
    else:
        # Input is treated directly as LR
        lr_rgb = img
        lr_yiq = rgb2yiq(lr_rgb)
        lr_y = lr_yiq[:, :, 0]
        lr_i = lr_yiq[:, :, 1]
        lr_q = lr_yiq[:, :, 2]
    
    # Save a copy of the original input for reference
    imsave_normalized(os.path.join(job_dir, "input.png"), np.clip(img, 0, 1))
    
    # Core SR Processing
    # Initialize solver with the Low-Resolution Luminance channel
    solver = SRSolver(lr_y)
    
    # Run the Unified SR algorithm
    # lambda_cl=30.0 enforces strong consistency with the input
    # lambda_ex=2.0 adds texture details from internal patch recurrence
    sr_y = solver.upscale(target_scale=float(target_scale), lambda_cl=30.0, lambda_ex=2.0, use_degradation=use_degradation)
    
    # Post-processing and color reconstruction
    h_sr, w_sr = sr_y.shape
    
    # Upscale Chrominance channels (I, Q) using standard Bicubic Interpolation
    sr_i = cv2.resize(lr_i, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    sr_q = cv2.resize(lr_q, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    
    # Merge channels and convert back to RGB
    sr_yiq = np.dstack((sr_y, sr_i, sr_q))
    sr_rgb = np.clip(yiq2rgb(sr_yiq), 0.0, 1.0)
    
    # Generate a pure Bicubic baseline for comparison
    bicubic_y = cv2.resize(lr_y, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    bicubic_yiq = np.dstack((bicubic_y, sr_i, sr_q))
    bicubic_rgb = np.clip(yiq2rgb(bicubic_yiq), 0.0, 1.0)
    
    # Save results
    out_bic = os.path.join(job_dir, "result_bicubic.png")
    out_ours = os.path.join(job_dir, "result_ours.png")
    imsave_normalized(out_bic, bicubic_rgb)
    imsave_normalized(out_ours, sr_rgb)
    
    # Metric calculation (Only if Ground Truth exists) ---
    if use_degradation:
        h_out, w_out, _ = sr_rgb.shape
        # Align Original HR to the exact output size
        hr_rgb_aligned = cv2.resize(hr_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)
        
        # Compute PSNR and SSIM
        p_bic = psnr(hr_rgb_aligned, bicubic_rgb, data_range=1.0)
        s_bic = ssim(hr_rgb_aligned, bicubic_rgb, data_range=1.0, channel_axis=2)
        p_our = psnr(hr_rgb_aligned, sr_rgb, data_range=1.0)
        s_our = ssim(hr_rgb_aligned, sr_rgb, data_range=1.0, channel_axis=2)
        
        metrics = {
            "bicubic": {"psnr": float(p_bic), "ssim": float(s_bic)},
            "ours": {"psnr": float(p_our), "ssim": float(s_our)},
        }
        
    return {
        "job_id": job_id,
        "use_degradation": use_degradation,
        "input": f"results/{job_id}/input.png",
        "lr_degraded": f"results/{job_id}/lr_degraded.png" if use_degradation else None,
        "bicubic": f"results/{job_id}/result_bicubic.png",
        "ours": f"results/{job_id}/result_ours.png",
        "metrics": metrics,
    }

def run_edge_based_sr(
    input_path: str,
    use_degradation: bool,
    target_scale: int = 2,
    use_ensemble: bool = True,  #  Hız/Kalite dengesi için
    out_dir: str = "static/results"
):
    """
    Pipeline to run the edge-based sr
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(out_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    downscale_factor = int(target_scale)
    
    # Read image (0-1 Float RGB)
    img = imread_normalized(input_path)
    
    #  Crop Modulo (Dimension Alignment) 
    # If degradation is used, crop the image to multiples of the scale factor.
    # This ensures correct comparison with "Ground Truth" 
    if use_degradation:
        h, w, _ = img.shape
        h_new = h - (h % downscale_factor)
        w_new = w - (w % downscale_factor)
        img = img[:h_new, :w_new, :]
        
    hr_rgb = img
    metrics = None
    
    # Degradation logic
    if use_degradation:
        h, w, _ = hr_rgb.shape
        lr_h, lr_w = h // downscale_factor, w // downscale_factor
        
        # Simulation: High Res -> Low Res
        lr_rgb = cv2.resize(hr_rgb, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        imsave_normalized(os.path.join(job_dir, "lr_degraded.png"), np.clip(lr_rgb, 0, 1))
    else:
        # If no degradation, the input image is considered LR
        lr_rgb = img
        
    imsave_normalized(os.path.join(job_dir, "input.png"), np.clip(img, 0, 1))
    #call the edge based sr
    solver = OptimizedEdgeSR(scale_factor=target_scale)
    # ensemble is True for better quality, False for faster processing
    sr_rgb = solver.upscale(lr_rgb, use_ensemble=use_ensemble)
    
    # bicubic for comparison making two images same size for fair comparision
    h_sr, w_sr, _ = sr_rgb.shape
    bicubic_rgb = cv2.resize(lr_rgb, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    
    out_bic = os.path.join(job_dir, "result_bicubic.png")
    out_ours = os.path.join(job_dir, "result_ours.png")
    
    imsave_normalized(out_bic, bicubic_rgb)
    imsave_normalized(out_ours, sr_rgb)
    
    # Metrik Hesaplama
    if use_degradation:
        #metrics on Y channel
        def to_y_channel(img_rgb):
            # from RGB to Y channel then return
            img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
            img_y = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)[:,:,0]
            return img_y.astype(np.float32) / 255.0

        hr_y = to_y_channel(hr_rgb)
        bic_y = to_y_channel(bicubic_rgb)
        sr_y = to_y_channel(sr_rgb)

        p_bic = psnr(hr_y, bic_y, data_range=1.0)
        s_bic = ssim(hr_y, bic_y, data_range=1.0) # channel_axis gerekmez, tek kanal
        
        p_our = psnr(hr_y, sr_y, data_range=1.0)
        s_our = ssim(hr_y, sr_y, data_range=1.0)
        
        metrics = {
            "bicubic": {"psnr": float(p_bic), "ssim": float(s_bic)},
            "ours": {"psnr": float(p_our), "ssim": float(s_our)},
        }
        
    return {
        "job_id": job_id,
        "use_degradation": use_degradation,
        "input": f"results/{job_id}/input.png",
        "lr_degraded": f"results/{job_id}/lr_degraded.png" if use_degradation else None,
        "bicubic": f"results/{job_id}/result_bicubic.png",
        "ours": f"results/{job_id}/result_ours.png",
        "metrics": metrics,
    }