import cv2
import numpy as np
from scipy.ndimage import shift
from src.patch_matcher import PatchMatcher, extract_patches
from src.pyramid import create_gaussian_pyramid

import cv2
import numpy as np

def back_projection(high_res,
                    low_res,
                    downscale_factor=2,
                    sigma=None,
                    iterations=10,
                    alpha=0.2):
    """
    Glasner tarzı observation term için back-projection:

      L ≈ D (H * G_sigma)

    Burada:
      - D: subsample (↓ downscale_factor)
      - G_sigma: Gaussian blur (sigma verilirse sabit, None ise s^2-1'dan türetilir)

    Algoritma (iteratif):
      1) current_hr'i blurla ve downsample et -> current_lr
      2) hata = low_res - current_lr
      3) hatayı upsample edip blur'la -> correction
      4) current_hr += alpha * correction
    """

    hr = high_res.astype(np.float32)
    lr = low_res.astype(np.float32)

    h_hr, w_hr = hr.shape
    h_lr, w_lr = lr.shape

    """# scale'ı boyuttan da tahmin et (güvenlik için)
    if h_lr > 0 and w_lr > 0:
        scale_y = h_hr / float(h_lr)
        scale_x = w_hr / float(w_lr)
        approx_scale = (scale_x + scale_y) * 0.5
        # integer değilse bile verilen downscale_factor'i kullanıyoruz
        # ama info için bu değeri elde tutmak faydalı
    else:
        approx_scale = float(downscale_factor)"""

    if sigma is None:
        # Glasner: sigma ~ sqrt(s^2 - 1)
        sigma = np.sqrt(downscale_factor ** 2 - 1.0)

    current_hr = hr.copy()

    for _ in range(iterations):
        # --- 1) Blur + downsample ---
        if sigma > 1e-6:
            blurred = cv2.GaussianBlur(
                current_hr, (0, 0),
                sigmaX=sigma, sigmaY=sigma
            )
        else:
            blurred = current_hr

        # Boyut tam olarak integer çarpan mı?
        if (h_hr == h_lr * downscale_factor) and (w_hr == w_lr * downscale_factor):
            # Tam Glasner: stride ile subsample
            current_lr = blurred[::int(downscale_factor), ::int(downscale_factor)]
        else:
            # Aradaki scale tam değilse, area resize ile yaklaşıkla
            current_lr = cv2.resize(
                blurred,
                (w_lr, h_lr),
                interpolation=cv2.INTER_AREA
            )

        # --- 2) LR domain'inde hata ---
        diff = lr - current_lr

        """# --- 3) Hatayı HR'e upsample et (D^T * G^T yaklaşık) ---
        if (h_hr == h_lr * downscale_factor) and (w_hr == w_lr * downscale_factor):
            # Önce zeros-insertion, sonra blur (adjoint operatöre yakın)
            up = np.zeros_like(current_hr, dtype=np.float32)
            up[::downscale_factor, ::downscale_factor] = diff

            if sigma > 1e-6:
                up = cv2.GaussianBlur(
                    up, (0, 0),
                    sigmaX=sigma, sigmaY=sigma
                )
        else:
            # Non-integer scale -> basit bicubic upsample (yaklaşık)
            up = cv2.resize(
                diff,
                (w_hr, h_hr),
                interpolation=cv2.INTER_CUBIC
            )

        # --- 4) HR tahminini güncelle ---
        current_hr = current_hr + alpha * up

    return np.clip(current_hr, 0.0, 1.0)"""
    
        up = cv2.resize(
            diff,
            (w_hr, h_hr),
            interpolation=cv2.INTER_CUBIC
        )
        
        correction = alpha*up
        limit = 0.03
        correction = np.clip(correction, -limit, limit)
        #up = cv2.GaussianBlur(up, (0, 0), sigmaX=0.5, sigmaY=0.5)
        #current_hr = current_hr + alpha * up
        current_hr = current_hr + correction
        
    return np.clip(current_hr, 0.0, 1.0)


class SRSolver:
    def __init__(self, input_image, scale_factor=1.25):
        self.input_image = input_image
        self.scale_factor = scale_factor
        
        #create the image pyramid (this will be our database)
        self.pyramid = create_gaussian_pyramid(self.input_image, scale_factor=scale_factor, max_depth=6)
        
        self.matchers = []
        for lvl in range(1, len(self.pyramid)):
            img_lvl = self.pyramid[lvl]
            matcher = PatchMatcher(img_lvl, patch_size=5)
            self.matchers.append(matcher)
        
        self.matcher_level_indices = list(range(1, len(self.pyramid)))    
        
    def enhance_details(self, upscaled_img):
        h, w = upscaled_img.shape
        patch_size = 5
        stride = 1
        
        reconstruction = np.zeros_like(upscaled_img, dtype=np.float32)
        weight_map = np.zeros_like(upscaled_img, dtype=np.float32)
        
        total_patches = 0
        passed_patches = 0
        
        print(f"DEBUG: Max pixel value: {upscaled_img.max():.4f}")
        
        all_patches, _ = extract_patches(upscaled_img, patch_size)
        all_stds = np.std(all_patches, axis=1)
        
        variance_threshold = np.percentile(all_stds, 90)
        variance_threshold = max(variance_threshold, 0.005)
        print(f"Dynamic Variance Threshold: {variance_threshold:.5f}")
        
        sample_dists = []
        sample_stride = 8
        if len(self.matchers) > 0:
            base_matcher = self.matchers[0]
            for y in range(0, h - patch_size + 1, sample_stride):
                for x in range(0, w - patch_size + 1, sample_stride):
                    patch = upscaled_img[y:y+patch_size, x:x+patch_size]
                    if np.std(patch) < variance_threshold:
                        continue
                    _, _, d = base_matcher.find_nearest_neighbors(patch)
                    sample_dists.append(d)
                    if len(sample_dists) >= 500:
                        break
                if len(sample_dists) >= 500:
                    break
            
        if len(sample_dists) > 0:
            median_dist = float(np.median(sample_dists))
        else:
            median_dist = 1e-3
            
        print(f"Sampled median NN distance: {median_dist:.6f}")
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                
                total_patches += 1
                
                current_patch = upscaled_img[y:y+patch_size, x:x+patch_size]
                
                if np.std(current_patch) < variance_threshold:
                    continue
                
                multi_detail_num = np.zeros((patch_size, patch_size), dtype=np.float32)
                multi_detail_den = 0.0
                
                for idx, matcher in enumerate(self.matchers):
                    level = self.matcher_level_indices[idx]
                    coarse_img = self.pyramid[level]
                    parent_img = self.pyramid[level - 1]
                    
                    patch_centered = current_patch - current_patch.mean()
                    shifted_patch = np.roll(patch_centered, shift=1, axis=0)
                    self_dist = np.linalg.norm(patch_centered - shifted_patch)
                    self_dist = max(self_dist, 1e-6)
                    
                    match_patch, (my, mx), dist = matcher.find_nearest_neighbors(current_patch)
                
                    dist_threshold = min(0.7 * self_dist, 1.2 * median_dist)
                    if dist > dist_threshold:
                        continue
                
                    py = int(my * self.scale_factor)
                    px = int(mx * self.scale_factor)
                
                    if py + patch_size > parent_img.shape[0] or px + patch_size > parent_img.shape[1]:
                        continue
                
                    parent_patch = parent_img[py:py+patch_size, px:px+patch_size]
                
                
                    if median_dist > 0:
                        normalized_dist = dist / median_dist
                    else:
                        normalized_dist = dist
                    
                    beta = 1.5
                    weight = np.exp(-beta * normalized_dist)
                
                    if weight < 0.35:
                        continue
                
                    detail = parent_patch - match_patch
                    
                    multi_detail_num += weight * detail
                    multi_detail_den += weight
                
                if multi_detail_den <= 0:
                    continue
                
                mean_detail = multi_detail_num / multi_detail_den
                
                detail_scale = 0.4
                target_patch = current_patch + detail_scale * mean_detail
                
                patch_weight = multi_detail_den
                
                reconstruction[y:y+patch_size, x:x+patch_size] += target_patch * patch_weight
                weight_map[y:y+patch_size, x:x+patch_size] += patch_weight
                
                passed_patches += 1
        
        ratio = (passed_patches / total_patches) * 100 if total_patches > 0 else 0
        print(f"  -> Patch Stats: {passed_patches}/{total_patches} passed (Ratio: {ratio:.2f}%)")
        
        final_img = upscaled_img.copy()
        mask = weight_map > 0.0
        final_img[mask] = reconstruction[mask] / weight_map[mask]
        
        return final_img
    
    def enhance_details_ls(self, upscaled_img):
        h, w = upscaled_img.shape
        patch_size = 5
        stride = 1
        
        num_ex = np.zeros_like(upscaled_img, dtype=np.float32)
        den_ex = np.zeros_like(upscaled_img, dtype=np.float32)
        
        total_patches = 0
        used_patches = 0
        
        print(f"DEBUG: Max pixel value: {upscaled_img.max():.4f}")
        
        all_patches, _ = extract_patches(upscaled_img, patch_size)
        all_stds = np.std(all_patches, axis=1)
        
        variance_threshold = np.percentile(all_stds, 90)
        variance_threshold = max(variance_threshold, 0.005)
        print(f"Dynamic Variance Threshold: {variance_threshold:.5f}")
        
        sample_dists = []
        sample_stride = 8
        if len(self.matchers) > 0:
            base_matcher = self.matchers[0]
            for y in range(0, h - patch_size + 1, sample_stride):
                for x in range(0, w - patch_size + 1, sample_stride):
                    patch = upscaled_img[y:y+patch_size, x:x+patch_size]
                    if np.std(patch) < variance_threshold:
                        continue
                    _, _, d = base_matcher.find_nearest_neighbors(patch)
                    sample_dists.append(d)
                    if len(sample_dists) >= 500:
                        break
                if len(sample_dists) >= 500:
                    break
                
        median_dist = float(np.median(sample_dists)) if sample_dists else 1e-3
        print(f"Sampled median NN distance: {median_dist:.6f}")
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):

                total_patches += 1

                current_patch = upscaled_img[y:y+patch_size, x:x+patch_size]

                # Düşük varyanslı (düz) patch'leri kısıtlara eklemiyoruz
                if np.std(current_patch) < variance_threshold:
                    continue

                # Bu patch için tüm seviyelerden gelen detayların ortalaması
                multi_detail_num = np.zeros((patch_size, patch_size), dtype=np.float32)
                multi_detail_den = 0.0

                for idx, matcher in enumerate(self.matchers):
                    level      = self.matcher_level_indices[idx]   # 1,2,3,...
                    coarse_img = self.pyramid[level]
                    parent_img = self.pyramid[level - 1]

                    # self-distance (lokal noise floor)
                    patch_centered = current_patch - current_patch.mean()
                    shifted_patch  = np.roll(patch_centered, shift=1, axis=0)
                    self_dist      = np.linalg.norm(patch_centered - shifted_patch)
                    self_dist      = max(self_dist, 1e-6)

                    # en yakın komşu
                    match_patch, (my, mx), dist = matcher.find_nearest_neighbors(current_patch)

                    # mesafe eşiği: self_dist ve global medyanın birleşimi
                    dist_threshold = min(0.7 * self_dist, 1.2 * median_dist)
                    if dist > dist_threshold:
                        continue

                    # parent patch (bir üst seviye)
                    py = int(my * self.scale_factor)
                    px = int(mx * self.scale_factor)
                    if (py + patch_size > parent_img.shape[0] or
                        px + patch_size > parent_img.shape[1]):
                        continue

                    parent_patch = parent_img[py:py+patch_size, px:px+patch_size]

                    # similarity tabanlı weight
                    normalized_dist = dist / median_dist if median_dist > 0 else dist
                    beta   = 1.5
                    weight = np.exp(-beta * normalized_dist)
                    if weight < 0.15:
                        continue

                    # İstersen coarse seviyeleri cezalandır:
                    # level_penalty = 1.0 / (1.0 + 0.3 * (level - 1))
                    # weight *= level_penalty

                    # Bu seviyeden gelen detay
                    detail = parent_patch - match_patch

                    multi_detail_num += weight * detail
                    multi_detail_den += weight

                # Bu patch için güvenilir multi-scale detay yoksa kısıt ekleme
                if multi_detail_den <= 0:
                    continue

                # Patch bazlı mean detail
                mean_detail = multi_detail_num / multi_detail_den

                # y_k: hedef patch (current_patch + detay)
                detail_scale = 0.25
                target_patch = current_patch + detail_scale * mean_detail  # y_k

                # w_k: bu patch'in toplam ağırlığı
                patch_weight = multi_detail_den

                # --------- Global LS normal denklemine katkı ---------
                num_ex[y:y+patch_size, x:x+patch_size] += patch_weight * target_patch
                den_ex[y:y+patch_size, x:x+patch_size] += patch_weight

                used_patches += 1

        ratio = (used_patches / total_patches) * 100 if total_patches > 0 else 0
        print(f"  -> Patch Stats (example-based): {used_patches}/{total_patches} used (Ratio: {ratio:.2f}%)")

        return num_ex, den_ex
    
    def upscale(self, target_scale=2.0,
                lambda_ex=1.0, lambda_cl=1.0):
        """
        Coarse-to-fine SR (Glasner'a daha yakın):

        current_scale = 1.0'dan başla, her adımda:
          1) current_hr'yi step_factor ile bicubic büyüt
          2) Example-based LS num_ex, den_ex hesapla
          3) Classical SR back-projection ile H_bp (LR'ye göre tutarlılık)
          4) Piksel başına LS birleşimi ile yeni current_hr hesapla

        Step faktörü: self.scale_factor (örn: 1.25)
        Son adımda target_scale'e kadar ayarlanıyor.
        """

        lr = self.input_image.astype(np.float32)
        h_lr, w_lr = lr.shape

        # başlangıç HR = LR (current_scale = 1.0)
        current_hr = lr.copy()
        current_scale = 1.0

        step_id = 0
        
        downscale_factor = int(round(target_scale))
        bp_sigma = None

        while current_scale < target_scale - 1e-6:
            step_id += 1

            # Bu adımda büyüme faktörü
            remaining = target_scale / current_scale
            step_factor = min(self.scale_factor, remaining)
            new_scale = current_scale * step_factor

            h_hr = int(round(h_lr * new_scale))
            w_hr = int(round(w_lr * new_scale))
            
            is_last_step = (abs(new_scale - target_scale) < 1e-6)

            print(f"\n=== Coarse-to-fine Step {step_id}: "
                  f"scale {current_scale:.3f} -> {new_scale:.3f} "
                  f"({h_hr}x{w_hr}) ===")

            # 1) Bicubic ile bir önceki HR'den yeni HR'e çık
            bicubic = cv2.resize(
                current_hr,
                (w_hr, h_hr),
                interpolation=cv2.INTER_CUBIC
            )
            bicubic = np.clip(bicubic, 0.0, 1.0)

            # 2) Example-based LS (num_ex, den_ex)
            print("  -> Example-based LS")
            num_ex, den_ex = self.enhance_details_ls(bicubic)

            if is_last_step:
            # 3) Classical SR back-projection (hep orijinal LR'ye göre)
                print("  -> Classical SR back-projection (LAST STEP)")
                #hr_bp = back_projection(bicubic, lr, downscale_factor=downscale_factor, sigma=bp_sigma, iterations=20, alpha=1.0)
                #hr_bp = np.clip(hr_bp, 0.0, 1.0)
                
                #lam_ex_step = lambda_ex
                #lam_cl_step = lambda_cl
                ########2
                bp_iters=2
            else:
                print("  -> Classical SR back-projection (INTERMEDIATE STEP)")
                #hr_bp = bicubic  # classical term = identity
                #lam_ex_step = lambda_ex
                #lam_cl_step = 0.0
                ########5
                bp_iters = 5
                
            hr_bp = back_projection(
                bicubic,
                lr,
                downscale_factor=downscale_factor,
                sigma=bp_sigma,
                iterations=bp_iters,
                alpha=0.2,
            )
            hr_bp = np.clip(hr_bp, 0.0, 1.0)
            
            lam_ex_step = lambda_ex
            lam_cl_step = lambda_cl
            
            #tau = np.percentile(den_ex[den_ex > 0], 90)
            #den_ex_clipped = np.minimum(den_ex, tau)

            # 4) Birleşik LS çözümü
            num_total = lam_ex_step * num_ex + lam_cl_step * hr_bp
            den_total = lam_ex_step * den_ex + lam_cl_step
            
            updated = hr_bp.copy()
            mask = den_total > 0.0
            updated[mask] = num_total[mask] / den_total[mask]
            updated = np.clip(updated, 0.0, 1.0)

            current_hr = updated
            current_scale = new_scale

        return current_hr
