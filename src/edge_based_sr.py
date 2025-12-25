import cv2
import numpy as np

class OptimizedEdgeSR:
    """
    Ebrar
    Kenar tabanlı süper çözünürlük için optimize edilmiş sınıf.
    Yalnızca Y kanalını işler ve RGB görüntüler için destek sağlar.
    """
    def __init__(self, scale_factor=2):
        self.scale = int(scale_factor)

    def upscale(self, img, use_ensemble=False):
        # Girdi: 0-1 Float veya 0-255 Uint8 (RGB veya Gray)
        input_img = img.astype(np.float32)
        if input_img.max() > 1.0:
            input_img /= 255.0

        is_color = len(input_img.shape) == 3

        if is_color:
            result_uint8 = self._process_rgb(input_img, use_ensemble)
        else:
            result_uint8 = self._process_channel(input_img, use_ensemble)

        # Çıktı: Proje standardına uygun (0.0 - 1.0 Float)
        return result_uint8.astype(np.float32) / 255.0
    def _process_rgb(self, img, use_ensemble):
        h, w, _ = img.shape
        target_h, target_w = h * self.scale, w * self.scale

        # Proje RGB okuyor, OpenCV BGR sever. Dönüşüm yapıyoruz.
        img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lr_y, lr_cb, lr_cr = cv2.split(img_ycc)

        # 1. Y Kanalını işle (Bu fonksiyon uint8 döner)
        sr_y = self._process_channel(lr_y, use_ensemble)

        # 2. Renk kanallarını büyüt (Bunlar float32 kalır)
        sr_cb = cv2.resize(lr_cb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sr_cr = cv2.resize(lr_cr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # 3. DÜZELTME: Renk kanallarını da uint8 formatına çevir (sr_y ile aynı olsun)
        sr_cb = (np.clip(sr_cb, 0.0, 1.0) * 255.0).astype(np.uint8)
        sr_cr = (np.clip(sr_cr, 0.0, 1.0) * 255.0).astype(np.uint8)

        # 4. GÜVENLİK: Nadiren 1 piksellik kayma olabilir, boyutları zorla eşitle
        if sr_y.shape != sr_cb.shape:
             sr_y = cv2.resize(sr_y, (sr_cb.shape[1], sr_cb.shape[0]))

        # Artık hepsi uint8 ve aynı boyutta
        sr_merged = cv2.merge([sr_y, sr_cb, sr_cr])
        
        # YCrCb -> RGB
        sr_final = cv2.cvtColor(sr_merged, cv2.COLOR_YCrCb2RGB)

        # Zaten uint8 olduğu için direkt dönüyoruz (clip gereksiz ama zararsız)
        return sr_final
    def _process_channel(self, channel, use_ensemble):
        if use_ensemble:
            res = self._process_ensemble(channel)
        else:
            res = self._process_single(channel)
        return (np.clip(res, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _process_single(self, lr_y):
        h, w = lr_y.shape
        target_h, target_w = h * self.scale, w * self.scale
        
        bicubic_upscale = cv2.resize(lr_y, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        edge_mask = self._get_canny_mask(bicubic_upscale)
        shocked_image = self._shock_filter(bicubic_upscale, iterations=1, dt=0.05)
        sr_mixed = shocked_image * edge_mask + bicubic_upscale * (1.0 - edge_mask)
        sr_final = self._ibp_process(sr_mixed, lr_y, iterations=5)
        
        return sr_final

    def _process_ensemble(self, lr_y):
        augmented_results = []
        for k in range(8):
            img_aug = self._augment_img(lr_y, k)
            res_aug = self._process_single(img_aug)
            res_restored = self._augment_img_rev(res_aug, k)
            augmented_results.append(res_restored)
        return np.mean(augmented_results, axis=0)

    # --- Helperlar (Önceki kodun aynısı) ---
    def _get_canny_mask(self, img):
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        edges = cv2.Canny(img_uint8, 100, 200) 
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        mask = dilated_edges.astype(np.float32) / 255.0
        return cv2.GaussianBlur(mask, (3, 3), 0)

    def _shock_filter(self, img, iterations=1, dt=0.05):
        img_shock = img.copy()
        for _ in range(iterations):
            sobelx = cv2.Sobel(img_shock, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_shock, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            laplacian = cv2.Laplacian(img_shock, cv2.CV_32F, ksize=3)
            sign_lap = -np.sign(laplacian)
            img_shock = img_shock + dt * sign_lap * magnitude
        return np.clip(img_shock, 0.0, 1.0)

    def _ibp_process(self, current_sr, original_lr, iterations=5):
        h_lr, w_lr = original_lr.shape
        img_t = current_sr.copy()
        for _ in range(iterations):
            downsampled = cv2.resize(img_t, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
            error = original_lr - downsampled
            error_upsampled = cv2.resize(error, (img_t.shape[1], img_t.shape[0]), interpolation=cv2.INTER_CUBIC)
            img_t = img_t + error_upsampled
            img_t = np.clip(img_t, 0.0, 1.0)
        return img_t

    def _augment_img(self, img, mode):
        if mode == 0: return img
        if mode == 1: return np.rot90(img, 1)
        if mode == 2: return np.rot90(img, 2)
        if mode == 3: return np.rot90(img, 3)
        if mode == 4: return np.flipud(img)
        if mode == 5: return np.rot90(np.flipud(img), 1)
        if mode == 6: return np.rot90(np.flipud(img), 2)
        if mode == 7: return np.rot90(np.flipud(img), 3)

    def _augment_img_rev(self, img, mode):
        if mode == 0: return img
        if mode == 1: return np.rot90(img, 3)
        if mode == 2: return np.rot90(img, 2)
        if mode == 3: return np.rot90(img, 1)
        if mode == 4: return np.flipud(img)
        if mode == 5: return np.flipud(np.rot90(img, 3))
        if mode == 6: return np.flipud(np.rot90(img, 2))
        if mode == 7: return np.flipud(np.rot90(img, 1))