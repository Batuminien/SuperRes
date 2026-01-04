import cv2
import numpy as np
import os
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from src.utils import (
    imread_normalized,
    rgb2yiq,
    yiq2rgb,
    degrade_gaussian_subsample,
)
from src.sr_solver import SRSolver
from src.edge_based_sr import OptimizedEdgeSR

def to_y_channel(img_rgb):
    """RGB görüntüyü Y kanalına çevir (metrik hesaplama için)"""
    img_u8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    img_y = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb)[:,:,0]
    return img_y.astype(np.float32) / 255.0

def main():
    input_dir = "input/Set14/Set14"
    scales = [2.0, 3.0, 4.0]
    
    # Set5 klasöründeki tüm png dosyalarını bul
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_paths:
        print(f"Hata: {input_dir} klasöründe görüntü bulunamadı!")
        return
    
    print(f"\n{'='*80}")
    print(f"Set5 Benchmark - Edge Based SR")
    print(f"Toplam {len(image_paths)} görüntü, {len(scales)} scale")
    print(f"{'='*80}\n")
    
    # Her scale için ortalama metrikleri sakla
    results_by_scale = {scale: {'psnr': [], 'ssim': []} for scale in scales}
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n--- {img_name} ---")
        
        # HR görüntüsünü yükle
        gt_img = imread_normalized(img_path)
        h_orig, w_orig = gt_img.shape[:2]
        
        for target_scale in scales:
            # HR'yi scale'e göre kırp (modulo alignment)
            downscale_factor = int(target_scale)
            h_new = h_orig - (h_orig % downscale_factor)
            w_new = w_orig - (w_orig % downscale_factor)
            gt_cropped = gt_img[:h_new, :w_new, :]
            
            # LR üret (degrade)
            hr_yiq = rgb2yiq(gt_cropped)
            lr_y = degrade_gaussian_subsample(hr_yiq[:,:,0], scale=downscale_factor, sigma=None)
            h_lr, w_lr = lr_y.shape
            
            lr_i = cv2.resize(hr_yiq[:,:,1], (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
            lr_q = cv2.resize(hr_yiq[:,:,2], (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
            lr_rgb = yiq2rgb(np.dstack((lr_y, lr_i, lr_q)))
            lr_rgb = np.clip(lr_rgb, 0.0, 1.0)
            
            # SR uygula
            edge_solver = OptimizedEdgeSR(scale_factor=target_scale)
            sr_rgb = edge_solver.upscale(lr_rgb, use_ensemble=True)
            sr_rgb = np.clip(sr_rgb, 0.0, 1.0)
            
            # Boyut eşitleme (gerekirse)
            if sr_rgb.shape[:2] != gt_cropped.shape[:2]:
                sr_rgb = cv2.resize(sr_rgb, (gt_cropped.shape[1], gt_cropped.shape[0]), 
                                   interpolation=cv2.INTER_CUBIC)
            
            # Y kanalı üzerinden metrik hesapla
            hr_y = to_y_channel(gt_cropped)
            sr_y = to_y_channel(sr_rgb)
            
            psnr_val = psnr(hr_y, sr_y, data_range=1.0)
            ssim_val = ssim(hr_y, sr_y, data_range=1.0)
            
            results_by_scale[target_scale]['psnr'].append(psnr_val)
            results_by_scale[target_scale]['ssim'].append(ssim_val)
            
            print(f"  x{int(target_scale)}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
    
    # Ortalama sonuçları yazdır
    print(f"\n{'='*80}")
    print(f"ORTALAMA SONUÇLAR (Set5)")
    print(f"{'='*80}")
    for scale in scales:
        avg_psnr = np.mean(results_by_scale[scale]['psnr'])
        avg_ssim = np.mean(results_by_scale[scale]['ssim'])
        print(f"x{int(scale)}: PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.4f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()