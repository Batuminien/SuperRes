import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from src.utils import (
    imread_normalized,
    rgb2yiq,
    yiq2rgb,
    imsave_normalized,
    degrade_gaussian_subsample,   # <-- yeni import
)
from src.sr_solver import SRSolver


def main():
    # settings
    use_degradation = False
    input_path = "input/Set14/Set14/bridge.png"
    target_scale = 2.0
    downscale_factor = int(target_scale)  # x2 SR için 2

    print(f"Operation is starting: {input_path}")
    print(f"Target Scale: x{target_scale}")

    # --- 1) HR görüntüyü yükle ---
    #print("Ground Truth image is loading...")
    img = imread_normalized(input_path)   # 0-1, HxWx3
    #H_in, W_in, C = img.shape
    hr_rgb = img
    print(f"Input Image Shape: {img.shape}")
    
    if use_degradation:
        print("Mode: HR -> blur + downsample -> LR")
        
        #hr_rgb = img
        print(f"Original HR Shape: {hr_rgb.shape}")

        # --- 2) HR'yi YIQ'a çevir, Y kanalını ayır ---
        hr_yiq = rgb2yiq(hr_rgb)
        hr_y = hr_yiq[:, :, 0]
        hr_i = hr_yiq[:, :, 1]
        hr_q = hr_yiq[:, :, 2]

        # --- 3) Glasner tarzı degrade ile LR Y üret ---
        # L = (H * G_sigma) ↓ scale
        sigma = None  # istersen sabit de verebilirsin, örn. sigma = 1.0
        lr_y = degrade_gaussian_subsample(hr_y, scale=downscale_factor, sigma=sigma)
        h_lr, w_lr = lr_y.shape

        # I ve Q kanallarını da LR boyutuna indir (basit bicubic)
        lr_i = cv2.resize(hr_i, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        lr_q = cv2.resize(hr_q, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)

        # LR'yi YIQ ve RGB olarak da saklayalım (görselleştirme için)
        lr_yiq = np.dstack((lr_y, lr_i, lr_q))
        lr_rgb = yiq2rgb(lr_yiq)

    #print(f"Low Resolution Shape (LR): {lr_rgb.shape}")
    
    else:
        print("Mode: Direct SR from LR input")
        lr_rgb = img
        h_lr, w_lr, _ = lr_rgb.shape
        
        lr_yiq = rgb2yiq(lr_rgb)
        lr_y = lr_yiq[:, :, 0]
        lr_i = lr_yiq[:, :, 1]
        lr_q = lr_yiq[:, :, 2]
        
    print(f"LR Shape: {lr_rgb.shape}")
        
        

    # --- 4) Super-resolution sadece Y kanalında ---
    solver = SRSolver(lr_y)  # scale_factor ve blur_sigma'yı istersen buradan ayarlarsın
    sr_y = solver.upscale(target_scale=target_scale,
                          lambda_cl=30.0,
                          lambda_ex=2.0)

    # --- 5) Renk kanallarını bicubic ile HR boyutuna çıkar ---
    h_sr, w_sr = sr_y.shape  # hedef: H, W ile aynı olmalı
    sr_i = cv2.resize(lr_i, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    sr_q = cv2.resize(lr_q, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)

    # YIQ -> RGB: SR sonucu
    sr_yiq = np.dstack((sr_y, sr_i, sr_q))
    sr_rgb = yiq2rgb(sr_yiq)

    # --- 6) Bicubic karşılaştırma (LR Y'den HR Y'ye) ---
    bicubic_y = cv2.resize(lr_y, (w_sr, h_sr), interpolation=cv2.INTER_CUBIC)
    bicubic_i = sr_i.copy()  # renk için aynı i/q kullanabiliriz
    bicubic_q = sr_q.copy()
    bicubic_yiq = np.dstack((bicubic_y, bicubic_i, bicubic_q))
    bicubic_rgb = yiq2rgb(bicubic_yiq)

    # --- 7) Clipping, kaydetme, gösterme ---
    sr_rgb = np.clip(sr_rgb, 0.0, 1.0)
    bicubic_rgb = np.clip(bicubic_rgb, 0.0, 1.0)

    imsave_normalized("output/result_bicubic.png", bicubic_rgb)
    imsave_normalized("output/result_ours.png", sr_rgb)

    print("\nOperation is done.")
    
    title_bicubic = "Standard Bicubic"
    title_ours = "Our SR Method"
    
    if use_degradation:
        # Downscale işlemi sırasında integer yuvarlama yüzünden
        # orijinal HR ile SR sonucu arasında 1-2 piksellik fark olabilir.
        # Metrik hesaplamak için HR'yi SR boyutuna tam eşitleyelim (crop/resize).
        h_out, w_out, _ = sr_rgb.shape
        hr_rgb_aligned = cv2.resize(hr_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)

        # Bicubic Metrics
        p_bic = psnr(hr_rgb_aligned, bicubic_rgb, data_range=1.0)
        s_bic = ssim(hr_rgb_aligned, bicubic_rgb, data_range=1.0, channel_axis=2)

        # Our SR Metrics
        p_our = psnr(hr_rgb_aligned, sr_rgb, data_range=1.0)
        s_our = ssim(hr_rgb_aligned, sr_rgb, data_range=1.0, channel_axis=2)

        # Başlıkları güncelle
        title_bicubic += f"\nPSNR: {p_bic:.2f}dB / SSIM: {s_bic:.4f}"
        title_ours += f"\nPSNR: {p_our:.2f}dB / SSIM: {s_our:.4f}"

    plt.figure(figsize=(14, 7))

    # Bicubic result
    plt.subplot(1, 2, 1)
    plt.imshow(bicubic_rgb)
    plt.title(title_bicubic, fontsize=12, fontweight='bold')
    plt.axis("off")

    # Our result
    plt.subplot(1, 2, 2)
    plt.imshow(sr_rgb)
    plt.title(title_ours, fontsize=12, fontweight='bold', color='darkblue')
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
