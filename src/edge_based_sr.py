import cv2
import numpy as np

class OptimizedEdgeSR:
    def __init__(self, scale_factor=2):
        self.scale = float(scale_factor) 

    def upscale(self, img, use_ensemble=False):
        
        input_img = img.astype(np.float32)
        if input_img.max() > 1.0:
            input_img /= 255.0

        is_color = len(input_img.shape) == 3
        if is_color:
            result_uint8 = self._process_rgb(input_img, use_ensemble)
        else:
            result_uint8 = self._process_channel(input_img, use_ensemble)

        return result_uint8.astype(np.float32) / 255.0

    def _process_rgb(self, img, use_ensemble):
        "PProcess RGB image by separating YCbCr channels."
        h, w, _ = img.shape
        target_h = int(round(h * self.scale))
        target_w = int(round(w * self.scale))

        img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        lr_y, lr_cb, lr_cr = cv2.split(img_ycc)

        sr_y = self._process_channel(lr_y, use_ensemble)

        # Lanczos4 for Cb, Cr
        sr_cb = cv2.resize(lr_cb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        sr_cr = cv2.resize(lr_cr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        sr_cb = (np.clip(sr_cb, 0.0, 1.0) * 255.0).astype(np.uint8)
        sr_cr = (np.clip(sr_cr, 0.0, 1.0) * 255.0).astype(np.uint8)

        if sr_y.shape != sr_cb.shape:
             sr_y = cv2.resize(sr_y, (sr_cb.shape[1], sr_cb.shape[0]))
        #sr_merged is now the YCrCb image
        sr_merged = cv2.merge([sr_y, sr_cb, sr_cr])
        sr_final = cv2.cvtColor(sr_merged, cv2.COLOR_YCrCb2RGB)
        return sr_final

    def _process_channel(self, channel, use_ensemble):
        if use_ensemble:
            res = self._process_ensemble(channel)
        else:
            res = self._process_single(channel)
        return (np.clip(res, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _process_single(self, lr_y):
        return self._process_single_step(lr_y, self.scale)

    def _process_single_step(self, original_lr, scale_factor):
        "For single channel image SR."
        h, w = original_lr.shape
        target_h = int(round(h * float(scale_factor)))
        target_w = int(round(w * float(scale_factor)))
        
        # upcale with Lanczos4
        bicubic_upscale = cv2.resize(original_lr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        #Coherence-Enhancing Shock Filter 
        iters = 2 if float(scale_factor) > 3.0 else 1
        shocked_image = self._coherence_shock_filter(
            bicubic_upscale,
            sigma=0.8,
            rho=2.5,
            dt=0.05,
            iterations=iters,
        )

        # canny edge mask
        edge_mask = self._get_canny_mask(bicubic_upscale)

        # mixing for edges canny, else bicubic
        sr_mixed = shocked_image * edge_mask + bicubic_upscale * (1.0 - edge_mask)

        # balance with IBP
        sr_final = self._ibp_process(sr_mixed, original_lr, iterations=12)

        # additional Laplacian sharpening for large scales
        if float(scale_factor) > 3.0:
            sharpen_kernel = np.array([[0, -1,  0],
                                       [-1,  5, -1],
                                       [0, -1,  0]], dtype=np.float32)
            sr_final = cv2.filter2D(sr_final, -1, sharpen_kernel)

        return np.clip(sr_final, 0.0, 1.0)

    def _coherence_shock_filter(self, img, sigma=1.0, rho=2.0, dt=0.05, iterations=1):
        """Coherence-Enhancing Shock Filter (Weickert, 2003).

        uses directional second derivatives along flow lines to enhance edges
        """
        img_shock = img.copy()
        
        for _ in range(iterations):
            # gradients 
            sobelx = cv2.Sobel(img_shock, cv2.CV_32F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_shock, cv2.CV_32F, 0, 1, ksize=3)

            #  Structure tensor J = G_rho * (∇u ∇u^T)
            Jxx = cv2.GaussianBlur(sobelx**2, (0, 0), rho)
            Jyy = cv2.GaussianBlur(sobely**2, (0, 0), rho)
            Jxy = cv2.GaussianBlur(sobelx * sobely, (0, 0), rho)

            # flow direction teta
            theta = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
            c = np.cos(theta)
            s = np.sin(theta)

            # directed second derivative v_ww
            img_smooth = cv2.GaussianBlur(img_shock, (0, 0), sigma)
            u_xx = cv2.Sobel(img_smooth, cv2.CV_32F, 2, 0, ksize=3)
            u_yy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 2, ksize=3)
            u_xy = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 1, ksize=3)
            v_ww = (c**2) * u_xx + 2 * c * s * u_xy + (s**2) * u_yy

            # new shock u_t = -sign(v_ww) * |∇u|
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            shock_update = -np.sign(v_ww) * magnitude
            img_shock = img_shock + dt * shock_update

        return np.clip(img_shock, 0.0, 1.0)

    def _get_canny_mask(self, img):
        "Canny edge detection mask"
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        # NL-Means: h=5 okey çalışıyo
        denoised = cv2.fastNlMeansDenoising(img_uint8, None, h=5, templateWindowSize=7, searchWindowSize=23)
        v = np.median(denoised)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(denoised, lower, upper, apertureSize=5, L2gradient=True)
        kernel = np.ones((3,3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        mask = dilated_edges.astype(np.float32) / 255.0
        return cv2.GaussianBlur(mask, (3, 3), 1.0)

    def _shock_filter(self, img, iterations=1, dt=0.05):
        "Simple Shock Filter"
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
            error_upsampled = cv2.resize(error, (img_t.shape[1], img_t.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            img_t = img_t + error_upsampled
            img_t = np.clip(img_t, 0.0, 1.0)
        return img_t

    def _process_ensemble(self, lr_y):
        "Applies test-time augmentation (8-corners) ensemble."
        augmented_results = []
        for k in range(8):
            img_aug = self._augment_img(lr_y, k)
            res_aug = self._process_single(img_aug)
            res_restored = self._augment_img_rev(res_aug, k)
            augmented_results.append(res_restored)
        return np.mean(augmented_results, axis=0)

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