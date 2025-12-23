import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.utils import (
    imread_normalized,
    rgb2yiq,
    yiq2rgb,
    degrade_gaussian_subsample,  # <-- Glasner tarzı degrade fonksiyonu
)
from src.sr_solver import SRSolver


def calculate_metrics(original, reconstructed):
    """
    Calculates PSNR and SSIM between two images.
    Comparison is done on the Y (luminance) channel (SR runs there).
    """
    orig = np.clip(original, 0, 1)
    recon = np.clip(reconstructed, 0, 1)

    score_psnr = psnr(orig, recon, data_range=1.0)
    score_ssim = ssim(orig, recon, data_range=1.0)

    return score_psnr, score_ssim


def main():
    gt_path = "input/test.jpg"

    print(f"Ground Truth image is loading: {gt_path}")
    gt_img = imread_normalized(gt_path)   # 0-1, HxWx3

    # Boyutu 4'ün katına kırp (2x downscale + 2x upscale için güvenli)
    h, w, c = gt_img.shape
    h = (h // 4) * 4
    w = (w // 4) * 4
    gt_img = gt_img[:h, :w, :]

    # Y kanalını al
    gt_yiq = rgb2yiq(gt_img)
    gt_y = gt_yiq[:, :, 0]

    # --- Glasner tarzı sentetik low-resolution üretimi ---
    # L = (H * G_sigma) ↓ 2
    downscale_factor = 2
    print("Creating synthetic LR with Gaussian blur + subsample...")
    lr_y = degrade_gaussian_subsample(gt_y, scale=downscale_factor, sigma=None)
    lr_h, lr_w = lr_y.shape

    print(f"Original Shape: {gt_y.shape}")
    print(f"Low Resolution Shape: {lr_y.shape}")
    print("-" * 30)

    # --- Bicubic interpolation baseline (LR -> HR) ---
    print("Bicubic scaling is doing...")
    bicubic_y = cv2.resize(lr_y, (w, h), interpolation=cv2.INTER_CUBIC)

    # --- Our SR method (only on Y) ---
    print("SR Method is running...")
    solver = SRSolver(lr_y, scale_factor=1.25)
    our_y = solver.upscale(target_scale=2.0,
                           lambda_cl=1.0,
                           lambda_ex=0.5)

    # Her ihtimale karşı boyut uyumsuzsa yeniden boyutlandır
    if our_y.shape != (h, w):
        our_y = cv2.resize(our_y, (w, h), interpolation=cv2.INTER_CUBIC)

    # --- Metrikleri hesapla (Y kanalında) ---
    b_psnr, b_ssim = calculate_metrics(gt_y, bicubic_y)
    print(f"Bicubic -> PSNR: {b_psnr:.4f} dB | SSIM: {b_ssim:.4f}")

    o_psnr, o_ssim = calculate_metrics(gt_y, our_y)
    print(f"SR -> PSNR: {o_psnr:.4f} dB | SSIM: {o_ssim:.4f}")

    diff_psnr = o_psnr - b_psnr
    if diff_psnr > 0:
        print(f"\nOur method is better by {diff_psnr:.4f} dB.")
    else:
        print(f"\nSorry bro! I tried. (ΔPSNR = {diff_psnr:.4f} dB)")


if __name__ == "__main__":
    main()
