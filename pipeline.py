import os
import uuid
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

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
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(out_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    downscale_factor = int(target_scale)
    
    img = imread_normalized(input_path)
    hr_rgb = img
    metrics = None
    
    if use_degradation:
        hr_yiq = rgb2yiq(hr_rgb)
        hr_y = hr_yiq[:, :, 0]
        hr_i = hr_yiq[:, :, 1]
        hr_q = hr_yiq[:, :, 2]
        
        lr_y = degrade_gaussian_subsample(hr_y, scale=downscale_factor, sigma=None)
        h_lr, w_lr = lr_y.shape
        
        lr_i = cv2.resize(hr_i, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        lr_q = cv2.resize(hr_q, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        
        lr_yiq = np.dstack((lr_y, lr_i, lr_q))
        lr_rgb = yiq2rgb(lr_yiq)
        
        imsave_normalized(os.path.join(job_dir, "lr_degraded.png"), np.clip(lr_rgb, 0, 1))
    else:
        lr_rgb = img
        lr_yiq = rgb2yiq(lr_rgb)
        lr_y = lr_yiq[:, :, 0]
        lr_i = lr_yiq[:, :, 1]
        lr_q = lr_yiq[:, :, 2]
        
    imsave_normalized(os.path.join(job_dir, "input.png"), np.clip(img, 0, 1))
    
    solver = SRSolver(lr_y)
    sr_y = solver.upscale(target_scale=float(target_scale), lambda_cl=30.0, lambda_ex=2.0)
    
    h_sr, w_sr = sr_y.shape
    
    sr_i = cv2.resize(lr_i, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    sr_q = cv2.resize(lr_q, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    
    sr_yiq = np.dstack((sr_y, sr_i, sr_q))
    sr_rgb = np.clip(yiq2rgb(sr_yiq), 0.0, 1.0)
    
    bicubic_y = cv2.resize(lr_y, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    bicubic_yiq = np.dstack((bicubic_y, sr_i, sr_q))
    bicubic_rgb = np.clip(yiq2rgb(bicubic_yiq), 0.0, 1.0)
    
    out_bic = os.path.join(job_dir, "result_bicubic.png")
    out_ours = os.path.join(job_dir, "result_ours.png")
    imsave_normalized(out_bic, bicubic_rgb)
    imsave_normalized(out_ours, sr_rgb)
    
    if use_degradation:
        h_out, w_out, _ = sr_rgb.shape
        hr_rgb_aligned = cv2.resize(hr_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)
        
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