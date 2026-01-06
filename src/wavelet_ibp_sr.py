# src/wavelet_ibp_sr.py
import os
import numpy as np
import pywt
from PIL import Image
from skimage.filters import gaussian
from skimage.transform import resize
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# =========================
# Basic I/O + Resize utils
# =========================
def _to_rgb01(pil_img: Image.Image) -> np.ndarray:
    """PIL -> RGB float32 [0,1]"""
    return np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb01(img_rgb01: np.ndarray, out_path: str) -> None:
    """RGB float32 [0,1] -> PNG"""
    img_u8 = np.clip(img_rgb01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(out_path)


def bilinear_resize(img2d: np.ndarray, scale: int) -> np.ndarray:
    """2D bilinear resize using Pillow. img in [0,1]."""
    img = np.asarray(img2d, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)

    h, w = img.shape
    pil = Image.fromarray((img * 255.0).astype(np.uint8))
    pil_up = pil.resize((w * scale, h * scale), resample=Image.BILINEAR)
    return np.asarray(pil_up).astype(np.float32) / 255.0


def pad_to_even(img2d: np.ndarray) -> np.ndarray:
    """Pad a 2D image to have even H,W (needed by SWT)."""
    h, w = img2d.shape
    pad_h = 0 if (h % 2 == 0) else 1
    pad_w = 0 if (w % 2 == 0) else 1
    if pad_h or pad_w:
        img2d = np.pad(img2d, ((0, pad_h), (0, pad_w)), mode="edge")
    return img2d


def _resize_to_match(arr2d: np.ndarray, target_shape) -> np.ndarray:
    return resize(
        arr2d,
        output_shape=target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)


# =========================
# Color space
# =========================
def rgb_to_ycbcr(img_rgb: np.ndarray):
    """
    img_rgb: HxWx3 float32 [0,1]
    Returns: Y, Cb, Cr each HxW float32
    """
    img_rgb = np.clip(np.asarray(img_rgb, dtype=np.float32), 0.0, 1.0)
    R = img_rgb[..., 0]
    G = img_rgb[..., 1]
    B = img_rgb[..., 2]

    # ITU-R BT.601
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    """
    Inputs: Y, Cb, Cr each HxW
    Output: HxWx3 RGB float32 [0,1]
    """
    Y = np.asarray(Y, dtype=np.float32)
    Cb = np.asarray(Cb, dtype=np.float32)
    Cr = np.asarray(Cr, dtype=np.float32)

    Cb_shift = Cb - 0.5
    Cr_shift = Cr - 0.5

    R = Y + 1.402 * Cr_shift
    G = Y - 0.344136 * Cb_shift - 0.714136 * Cr_shift
    B = Y + 1.772 * Cb_shift

    img_rgb = np.stack([R, G, B], axis=-1)
    return np.clip(img_rgb, 0.0, 1.0).astype(np.float32)


# =========================
# Wavelet SR (GRAY, only scale=2)
# =========================
def wavelet_sr_hybrid_gray(img_lr: np.ndarray,
                           scale: int = 2,
                           wavelet: str = "db2",
                           beta: float = 0.8,
                           thresh: float = 0.01) -> np.ndarray:
    """
    Single-level edge-guided wavelet SR for grayscale image.
    (Supports only scale=2)
    """
    if scale != 2:
        raise ValueError("wavelet_sr_hybrid_gray supports only scale=2")

    img_lr = np.clip(np.asarray(img_lr, dtype=np.float32), 0.0, 1.0)

    # 1) Upsample (target size)
    img_up = bilinear_resize(img_lr, scale=scale)
    Ht, Wt = img_up.shape

    # 2) Pad for SWT safety
    img_up_pad = pad_to_even(img_up)

    # 3) DWT on padded
    cA_d, (cH_d, cV_d, cD_d) = pywt.dwt2(img_up_pad, wavelet)

    # 4) SWT level=1 on padded
    try:
        cA_s, (cH_s, cV_s, cD_s) = pywt.swt2(img_up_pad, wavelet, level=1)[0]
    except ValueError:
        img_up_pad2 = np.pad(img_up_pad, ((0, 2), (0, 2)), mode="edge")
        cA_d, (cH_d, cV_d, cD_d) = pywt.dwt2(img_up_pad2, wavelet)
        cA_s, (cH_s, cV_s, cD_s) = pywt.swt2(img_up_pad2, wavelet, level=1)[0]
        img_up_pad = img_up_pad2

    # 5) Match SWT HF sizes to DWT HF sizes
    target = cH_d.shape
    if cH_s.shape != target:
        cH_s = _resize_to_match(cH_s, target)
    if cV_s.shape != target:
        cV_s = _resize_to_match(cV_s, target)
    if cD_s.shape != target:
        cD_s = _resize_to_match(cD_s, target)

    # 6) Edge map from SWT bands
    E = np.sqrt(cH_s**2 + cV_s**2 + cD_s**2)
    E = E / (E.max() + 1e-8)

    # 7) Soft-threshold HF from DWT
    def soft(x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

    cH_d = soft(cH_d, thresh)
    cV_d = soft(cV_d, thresh)
    cD_d = soft(cD_d, thresh)

    # 8) Edge-guided gain
    gain = 1.0 + beta * E
    cH = gain * cH_d
    cV = gain * cV_d
    cD = gain * cD_d

    # 9) Reconstruct via IDWT
    img_sr_pad = pywt.idwt2((cA_d, (cH, cV, cD)), wavelet)

    # 10) Crop back to target (no pad)
    img_sr = img_sr_pad[:Ht, :Wt]
    return np.clip(img_sr, 0.0, 1.0).astype(np.float32)


# =========================
# IBP on Y channel (integer scale)
# =========================
def iterative_back_projection_Y(Y_sr: np.ndarray,
                                Y_lr: np.ndarray,
                                scale: int,
                                n_iters: int = 10,
                                alpha: float = 0.2,
                                sigma: float = 1.0) -> np.ndarray:
    """
    Integer-scale IBP on luminance.
    Uses slicing [::scale, ::scale] for downsample (works for 2/3/4).
    """
    Y = np.clip(np.asarray(Y_sr, dtype=np.float32), 0.0, 1.0)
    Y_lr = np.clip(np.asarray(Y_lr, dtype=np.float32), 0.0, 1.0)

    for _ in range(n_iters):
        Y_blur = gaussian(Y, sigma=sigma, preserve_range=True)
        Y_down = Y_blur[::scale, ::scale]
        err_lr = Y_lr - Y_down
        err_up = bilinear_resize(err_lr, scale)
        Y = np.clip(Y + alpha * err_up, 0.0, 1.0)

    return Y.astype(np.float32)


def ibp_general(Y_sr: np.ndarray,
                Y_lr: np.ndarray,
                n_iters: int = 10,
                alpha: float = 0.2,
                sigma: float = 1.0) -> np.ndarray:
    """
    General IBP that matches LR size via cv2.resize (works for any scale).
    """
    H, W = Y_sr.shape
    h, w = Y_lr.shape
    Y = Y_sr.astype(np.float32).copy()

    for _ in range(n_iters):
        Y_blur = gaussian(Y, sigma=sigma, preserve_range=True)
        Y_down = cv2.resize(Y_blur, (w, h), interpolation=cv2.INTER_AREA)
        err_lr = (Y_lr - Y_down).astype(np.float32)
        err_up = cv2.resize(err_lr, (W, H), interpolation=cv2.INTER_LINEAR)
        Y = Y + alpha * err_up
        Y = np.clip(Y, 0.0, 1.0)

    return Y.astype(np.float32)


# =========================
# Wavelet + IBP (COLOR)
# =========================
def wavelet_ibp_sr_color_x2(img_lr_rgb: np.ndarray,
                            wavelet: str = "db2",
                            beta: float = 0.8,
                            thresh: float = 0.01,
                            ibp_iters: int = 10,
                            ibp_alpha: float = 0.2,
                            ibp_sigma: float = 1.0) -> np.ndarray:
    """
    Core method for x2 only:
      - RGB -> YCbCr
      - wavelet SR on Y (x2)
      - IBP on Y (x2)
      - bilinear upscale Cb/Cr (x2)
    """
    img_lr_rgb = np.clip(np.asarray(img_lr_rgb, dtype=np.float32), 0.0, 1.0)
    Y, Cb, Cr = rgb_to_ycbcr(img_lr_rgb)

    Y_sr = wavelet_sr_hybrid_gray(Y, scale=2, wavelet=wavelet, beta=beta, thresh=thresh)
    Y_sr = iterative_back_projection_Y(
        Y_sr, Y, scale=2, n_iters=ibp_iters, alpha=ibp_alpha, sigma=ibp_sigma
    )

    Cb_sr = bilinear_resize(Cb, 2)
    Cr_sr = bilinear_resize(Cr, 2)
    return ycbcr_to_rgb(Y_sr, Cb_sr, Cr_sr)


def wavelet_ibp_x3(lr_rgb: np.ndarray,
                   wavelet: str = "db2",
                   beta: float = 0.8,
                   thresh: float = 0.01,
                   ibp_iters: int = 10,
                   ibp_alpha: float = 0.2,
                   ibp_sigma: float = 1.0) -> np.ndarray:
    """
    Practical x3:
      1) x2 wavelet+ibp
      2) resize x2->x3 with bicubic
      3) general IBP on Y to enforce LR consistency
    """
    # 1) x2
    sr2 = wavelet_ibp_sr_color_x2(
        lr_rgb, wavelet=wavelet, beta=beta, thresh=thresh,
        ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
    )

    h, w, _ = lr_rgb.shape
    target_h, target_w = h * 3, w * 3

    # 2) x2 -> x3
    sr2_u8 = (np.clip(sr2, 0, 1) * 255).astype(np.uint8)
    sr3_u8 = cv2.resize(sr2_u8, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sr3 = sr3_u8.astype(np.float32) / 255.0

    # 3) IBP general on Y
    Y_lr, Cb_lr, Cr_lr = rgb_to_ycbcr(lr_rgb)
    Y_sr, _, _ = rgb_to_ycbcr(sr3)

    Y_refined = ibp_general(Y_sr, Y_lr, n_iters=ibp_iters, alpha=ibp_alpha, sigma=ibp_sigma)

    Cb_3 = cv2.resize(Cb_lr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    Cr_3 = cv2.resize(Cr_lr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return ycbcr_to_rgb(Y_refined, Cb_3, Cr_3)


def wavelet_ibp_x4(lr_rgb: np.ndarray,
                   wavelet: str = "db2",
                   beta: float = 0.8,
                   thresh: float = 0.01,
                   ibp_iters: int = 10,
                   ibp_alpha: float = 0.2,
                   ibp_sigma: float = 1.0) -> np.ndarray:
    """
    x4 = apply x2 twice (each stage has wavelet+IBP on Y)
    """
    sr2 = wavelet_ibp_sr_color_x2(
        lr_rgb, wavelet=wavelet, beta=beta, thresh=thresh,
        ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
    )
    sr4 = wavelet_ibp_sr_color_x2(
        sr2, wavelet=wavelet, beta=beta, thresh=thresh,
        ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
    )
    return sr4


def wavelet_ibp_sr_color(img_lr_rgb: np.ndarray,
                         scale: int = 2,
                         wavelet: str = "db2",
                         beta: float = 0.8,
                         thresh: float = 0.01,
                         ibp_iters: int = 10,
                         ibp_alpha: float = 0.2,
                         ibp_sigma: float = 1.0) -> np.ndarray:
    """
    Dispatcher for x2/x3/x4
    """
    scale = int(scale)
    if scale == 2:
        return wavelet_ibp_sr_color_x2(
            img_lr_rgb, wavelet=wavelet, beta=beta, thresh=thresh,
            ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
        )
    if scale == 3:
        return wavelet_ibp_x3(
            img_lr_rgb, wavelet=wavelet, beta=beta, thresh=thresh,
            ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
        )
    if scale == 4:
        return wavelet_ibp_x4(
            img_lr_rgb, wavelet=wavelet, beta=beta, thresh=thresh,
            ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
        )
    raise ValueError("Supported scales for Wavelet+IBP: 2, 3, 4")


# =========================
# Web / Pipeline entrypoint
# =========================
def run_wavelet_ibp_sr(
    input_path: str,
    use_degradation: bool,
    target_scale: int = 2,
    out_dir: str = "static/results",
    wavelet: str = "db2",
    beta: float = 0.8,
    thresh: float = 0.01,
    ibp_iters: int = 10,
    ibp_alpha: float = 0.2,
    ibp_sigma: float = 1.0,
):
    """
    Matches Ebrar/Batuhan pipeline output format:
    saves input / (optional) lr_degraded / bicubic / ours and returns dict.
    """
    import uuid

    scale = int(target_scale)
    if scale not in (2, 3, 4):
        scale = 2

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(out_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # ---- Read input ----
    in_pil = Image.open(input_path).convert("RGB")
    img_rgb = _to_rgb01(in_pil)

    # ---- Decide LR/HR depending on metric mode ----
    hr_rgb = None
    lr_rgb = img_rgb.copy()
    metrics = None

    if use_degradation:
        hr_rgb = img_rgb

        # crop HR to be divisible by scale
        H, W, _ = hr_rgb.shape
        H2 = (H // scale) * scale
        W2 = (W // scale) * scale
        hr_rgb = hr_rgb[:H2, :W2, :]

        # simulate LR (blur + subsample)
        hr_blur = gaussian(hr_rgb, sigma=1.0, channel_axis=-1, preserve_range=True)
        hr_blur = np.clip(hr_blur, 0.0, 1.0)
        lr_rgb = hr_blur[::scale, ::scale, :]

        _save_rgb01(lr_rgb, os.path.join(job_dir, "lr_degraded.png"))

    _save_rgb01(img_rgb, os.path.join(job_dir, "input.png"))

    # ---- Bicubic baseline ----
    h_lr, w_lr, _ = lr_rgb.shape
    bic_u8 = cv2.resize(
        (lr_rgb * 255).astype(np.uint8),
        (w_lr * scale, h_lr * scale),
        interpolation=cv2.INTER_CUBIC
    )
    bic_rgb = bic_u8.astype(np.float32) / 255.0

    # ---- Our SR (x2/x3/x4 supported now) ----
    sr_rgb = wavelet_ibp_sr_color(
        lr_rgb,
        scale=scale,
        wavelet=wavelet,
        beta=beta,
        thresh=thresh,
        ibp_iters=ibp_iters,
        ibp_alpha=ibp_alpha,
        ibp_sigma=ibp_sigma
    )

    out_bic = os.path.join(job_dir, "result_bicubic.png")
    out_ours = os.path.join(job_dir, "result_ours.png")
    _save_rgb01(bic_rgb, out_bic)
    _save_rgb01(sr_rgb, out_ours)

    # ---- Metrics if HR exists ----
    if use_degradation and (hr_rgb is not None):
        h_out, w_out, _ = sr_rgb.shape
        hr_rgb_aligned = cv2.resize(hr_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)

        p_bic = psnr(hr_rgb_aligned, bic_rgb, data_range=1.0)
        s_bic = ssim(hr_rgb_aligned, bic_rgb, data_range=1.0, channel_axis=2)
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


# =========================
# Standalone quick test
# =========================
def main():
    import matplotlib.pyplot as plt

    hr_path = "kedi.jpg"
    scale = 3  # test 2/3/4 here
    sigma_lr = 1.0

    wavelet = "db2"
    beta = 0.8
    thresh = 0.01
    ibp_iters = 10
    ibp_alpha = 0.2
    ibp_sigma = 1.0

    hr_rgb = _to_rgb01(Image.open(hr_path).convert("RGB"))

    # crop divisible
    H, W, _ = hr_rgb.shape
    H2 = (H // scale) * scale
    W2 = (W // scale) * scale
    hr_rgb = hr_rgb[:H2, :W2, :]
    H, W, _ = hr_rgb.shape

    # simulate LR
    hr_blur = gaussian(hr_rgb, sigma=sigma_lr, channel_axis=-1, preserve_range=True)
    hr_blur = np.clip(hr_blur, 0.0, 1.0)
    lr_rgb = hr_blur[::scale, ::scale, :]

    # bicubic
    bic_u8 = cv2.resize((lr_rgb * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
    bic_rgb = bic_u8.astype(np.float32) / 255.0

    # wavelet+ibp
    sr_rgb = wavelet_ibp_sr_color(
        lr_rgb,
        scale=scale,
        wavelet=wavelet,
        beta=beta,
        thresh=thresh,
        ibp_iters=ibp_iters,
        ibp_alpha=ibp_alpha,
        ibp_sigma=ibp_sigma
    )

    print("PSNR bicubic:", psnr(hr_rgb, bic_rgb, data_range=1.0))
    print("PSNR ours   :", psnr(hr_rgb, sr_rgb, data_range=1.0))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(lr_rgb); plt.title(f"LR (sim)"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(bic_rgb); plt.title(f"Bicubic x{scale}"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(sr_rgb); plt.title(f"Wavelet+IBP x{scale}"); plt.axis("off")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()

# src/wavelet_ibp_sr.py
"""import os
import numpy as np
import pywt
from PIL import Image
from skimage.filters import gaussian
from skimage.transform import resize
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# =========================
# Basic I/O + Resize utils
# =========================
def _to_rgb01(pil_img: Image.Image) -> np.ndarray:
    
    return np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb01(img_rgb01: np.ndarray, out_path: str) -> None:

    img_u8 = np.clip(img_rgb01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(out_path)


def bilinear_resize(img2d: np.ndarray, scale: int) -> np.ndarray:
   
    img = np.asarray(img2d, dtype=np.float32)
    img = np.clip(img, 0.0, 1.0)

    h, w = img.shape
    pil = Image.fromarray((img * 255.0).astype(np.uint8))
    pil_up = pil.resize((w * scale, h * scale), resample=Image.BILINEAR)
    return np.asarray(pil_up).astype(np.float32) / 255.0


def pad_to_even(img2d: np.ndarray) -> np.ndarray:
    h, w = img2d.shape
    pad_h = 0 if (h % 2 == 0) else 1
    pad_w = 0 if (w % 2 == 0) else 1
    if pad_h or pad_w:
        img2d = np.pad(img2d, ((0, pad_h), (0, pad_w)), mode="edge")
    return img2d


def _resize_to_match(arr2d: np.ndarray, target_shape) -> np.ndarray:
    return resize(
        arr2d,
        output_shape=target_shape,
        order=1,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)


# =========================
# Color space
# =========================
def rgb_to_ycbcr(img_rgb: np.ndarray):
    
    img_rgb = np.clip(np.asarray(img_rgb, dtype=np.float32), 0.0, 1.0)
    R = img_rgb[..., 0]
    G = img_rgb[..., 1]
    B = img_rgb[..., 2]

    # ITU-R BT.601
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
 
    Y = np.asarray(Y, dtype=np.float32)
    Cb = np.asarray(Cb, dtype=np.float32)
    Cr = np.asarray(Cr, dtype=np.float32)

    Cb_shift = Cb - 0.5
    Cr_shift = Cr - 0.5

    R = Y + 1.402 * Cr_shift
    G = Y - 0.344136 * Cb_shift - 0.714136 * Cr_shift
    B = Y + 1.772 * Cb_shift

    img_rgb = np.stack([R, G, B], axis=-1)
    return np.clip(img_rgb, 0.0, 1.0).astype(np.float32)


# =========================
# Wavelet SR (GRAY, scale=2)
# =========================
def wavelet_sr_hybrid_gray(img_lr: np.ndarray,
                           scale: int = 2,
                           wavelet: str = "db2",
                           beta: float = 0.8,
                           thresh: float = 0.01) -> np.ndarray:
   
    if scale != 2:
        raise ValueError("Only scale=2 supported")

    img_lr = np.clip(np.asarray(img_lr, dtype=np.float32), 0.0, 1.0)

    # 1) Upsample (target size)
    img_up = bilinear_resize(img_lr, scale=scale)
    Ht, Wt = img_up.shape

    # 2) Pad for SWT safety
    img_up_pad = pad_to_even(img_up)

    # 3) DWT on padded
    cA_d, (cH_d, cV_d, cD_d) = pywt.dwt2(img_up_pad, wavelet)

    # 4) SWT level=1 on padded (may require even size)
    try:
        cA_s, (cH_s, cV_s, cD_s) = pywt.swt2(img_up_pad, wavelet, level=1)[0]
    except ValueError:
        # extra pad (rare)
        img_up_pad2 = np.pad(img_up_pad, ((0, 2), (0, 2)), mode="edge")
        cA_d, (cH_d, cV_d, cD_d) = pywt.dwt2(img_up_pad2, wavelet)
        cA_s, (cH_s, cV_s, cD_s) = pywt.swt2(img_up_pad2, wavelet, level=1)[0]
        img_up_pad = img_up_pad2  # keep consistent

    # 5) Match SWT HF sizes to DWT HF sizes
    target = cH_d.shape
    if cH_s.shape != target:
        cH_s = _resize_to_match(cH_s, target)
    if cV_s.shape != target:
        cV_s = _resize_to_match(cV_s, target)
    if cD_s.shape != target:
        cD_s = _resize_to_match(cD_s, target)

    # 6) Edge map from SWT bands
    E = np.sqrt(cH_s**2 + cV_s**2 + cD_s**2)
    E = E / (E.max() + 1e-8)

    # 7) Soft-threshold HF from DWT
    def soft(x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

    cH_d = soft(cH_d, thresh)
    cV_d = soft(cV_d, thresh)
    cD_d = soft(cD_d, thresh)

    # 8) Edge-guided gain
    gain = 1.0 + beta * E
    cH = gain * cH_d
    cV = gain * cV_d
    cD = gain * cD_d

    # 9) Reconstruct via IDWT
    img_sr_pad = pywt.idwt2((cA_d, (cH, cV, cD)), wavelet)

    # 10) Crop back to target (no pad)
    img_sr = img_sr_pad[:Ht, :Wt]
    return np.clip(img_sr, 0.0, 1.0).astype(np.float32)


# =========================
# IBP on Y channel
# =========================
def iterative_back_projection_Y(Y_sr: np.ndarray,
                                Y_lr: np.ndarray,
                                scale: int,
                                n_iters: int = 10,
                                alpha: float = 0.2,
                                sigma: float = 1.0) -> np.ndarray:
 
    Y = np.clip(np.asarray(Y_sr, dtype=np.float32), 0.0, 1.0)
    Y_lr = np.clip(np.asarray(Y_lr, dtype=np.float32), 0.0, 1.0)

    for _ in range(n_iters):
        Y_blur = gaussian(Y, sigma=sigma, preserve_range=True)
        Y_down = Y_blur[::scale, ::scale]
        err_lr = Y_lr - Y_down
        err_up = bilinear_resize(err_lr, scale)
        Y = Y + alpha * err_up
        Y = np.clip(Y, 0.0, 1.0)

    return Y.astype(np.float32)

def ibp_general(Y_sr, Y_lr, n_iters=10, alpha=0.2, sigma=1.0):
    
    H, W = Y_sr.shape
    h, w = Y_lr.shape
    Y = Y_sr.astype(np.float32).copy()

    for _ in range(n_iters):
        Y_blur = gaussian(Y, sigma=sigma, preserve_range=True)
        # HR -> LR simulate: resize down to LR size
        Y_down = cv2.resize(Y_blur, (w, h), interpolation=cv2.INTER_AREA)

        err_lr = (Y_lr - Y_down).astype(np.float32)

        # LR error -> HR grid
        err_up = cv2.resize(err_lr, (W, H), interpolation=cv2.INTER_LINEAR)
        Y += alpha * err_up

    return np.clip(Y, 0.0, 1.0).astype(np.float32)

# =========================
# Wavelet + IBP (COLOR)
# =========================
def wavelet_ibp_sr_color(img_lr_rgb: np.ndarray,
                         scale: int = 2,
                         wavelet: str = "db2",
                         beta: float = 0.8,
                         thresh: float = 0.01,
                         ibp_iters: int = 10,
                         ibp_alpha: float = 0.2,
                         ibp_sigma: float = 1.0) -> np.ndarray:
 
    img_lr_rgb = np.clip(np.asarray(img_lr_rgb, dtype=np.float32), 0.0, 1.0)
    Y, Cb, Cr = rgb_to_ycbcr(img_lr_rgb)

    Y_sr = wavelet_sr_hybrid_gray(Y, scale=scale, wavelet=wavelet, beta=beta, thresh=thresh)
    Y_sr = iterative_back_projection_Y(Y_sr, Y, scale=scale, n_iters=ibp_iters,
                                       alpha=ibp_alpha, sigma=ibp_sigma)

    Cb_sr = bilinear_resize(Cb, scale)
    Cr_sr = bilinear_resize(Cr, scale)
    return ycbcr_to_rgb(Y_sr, Cb_sr, Cr_sr)

def wavelet_ibp_x3(lr_rgb, wavelet="db2", beta=0.8, thresh=0.01,
                   ibp_iters=10, ibp_alpha=0.2, ibp_sigma=1.0):
    # 1) 2x wavelet+ibp
    sr2 = wavelet_ibp_sr_color(
        lr_rgb, scale=2, wavelet=wavelet, beta=beta, thresh=thresh,
        ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
    )

    # 2) 1.5x upscale (2x -> 3x): resize to target size
    h, w, _ = lr_rgb.shape
    target_h, target_w = h * 3, w * 3

    sr2_u8 = (np.clip(sr2,0,1)*255).astype(np.uint8)
    sr3_u8 = cv2.resize(sr2_u8, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    sr3 = sr3_u8.astype(np.float32) / 255.0

    Y_lr, Cb_lr, Cr_lr = rgb_to_ycbcr(lr_rgb)
    Y_sr, Cb_sr, Cr_sr = rgb_to_ycbcr(sr3)

    Y_refined = ibp_general(Y_sr, Y_lr, n_iters=ibp_iters, alpha=ibp_alpha, sigma=ibp_sigma)

    Cb_3 = cv2.resize(Cb_lr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    Cr_3 = cv2.resize(Cr_lr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return ycbcr_to_rgb(Y_refined, Cb_3, Cr_3)


def wavelet_ibp_x4(lr_rgb, wavelet="db2", beta=0.8, thresh=0.01,
                   ibp_iters=10, ibp_alpha=0.2, ibp_sigma=1.0):
    # 2x
    sr2 = wavelet_ibp_sr_color(
        lr_rgb, scale=2, wavelet=wavelet, beta=beta, thresh=thresh,
        ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
    )
    # bir kez daha 2x => 4x
    sr4 = wavelet_ibp_sr_color(
        sr2, scale=2, wavelet=wavelet, beta=beta, thresh=thresh,
        ibp_iters=ibp_iters, ibp_alpha=ibp_alpha, ibp_sigma=ibp_sigma
    )
    return sr4


# =========================
# Web / Pipeline entrypoint
# =========================
def run_wavelet_ibp_sr(
    input_path: str,
    use_degradation: bool,
    target_scale: int ,
    out_dir: str = "static/results",
    wavelet: str = "db2",
    beta: float = 0.8,
    thresh: float = 0.01,
    ibp_iters: int = 10,
    ibp_alpha: float = 0.2,
    ibp_sigma: float = 1.0,
):
    if target_scale == 2:
        sr_rgb = wavelet_ibp_sr_color(
            lr_rgb, scale=2, wavelet=wavelet,
            beta=beta,
            thresh=thresh,
            ibp_iters=ibp_iters,
            ibp_alpha=ibp_alpha,
            ibp_sigma=ibp_sigma)
    elif target_scale == 3:
        sr_rgb = wavelet_ibp_x3(lr_rgb, ...)
    elif target_scale == 4:
        sr_rgb = wavelet_ibp_x4(lr_rgb, ...)
    else:
        raise ValueError("Supported scales: 2,3,4")
    
    import uuid

    job_id = str(uuid.uuid4())[:8]
    job_dir = os.path.join(out_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    scale = int(target_scale)

    lr_pil = Image.open(input_path).convert("RGB")
    img_rgb = _to_rgb01(lr_pil)

    metrics = None

    if use_degradation:
        hr_rgb = img_rgb

        H, W, _ = hr_rgb.shape
        H2 = (H // scale) * scale
        W2 = (W // scale) * scale
        hr_rgb = hr_rgb[:H2, :W2, :]

        hr_blur = gaussian(hr_rgb, sigma=1.0, channel_axis=-1, preserve_range=True)
        hr_blur = np.clip(hr_blur, 0.0, 1.0)
        lr_rgb = hr_blur[::scale, ::scale, :]

        _save_rgb01(lr_rgb, os.path.join(job_dir, "lr_degraded.png"))

    else:
        lr_rgb = img_rgb
        hr_rgb = None

    _save_rgb01(img_rgb, os.path.join(job_dir, "input.png"))

    # Bicubic baseline
    lr_u8 = (np.clip(lr_rgb, 0, 1) * 255).astype(np.uint8)
    lr_pil2 = Image.fromarray(lr_u8)
    bic_pil = lr_pil2.resize((lr_pil2.width * scale, lr_pil2.height * scale), resample=Image.BICUBIC)
    bic_rgb = _to_rgb01(bic_pil)

    sr_rgb = wavelet_ibp_sr_color(
        lr_rgb,
        scale=scale,
        wavelet=wavelet,
        beta=beta,
        thresh=thresh,
        ibp_iters=ibp_iters,
        ibp_alpha=ibp_alpha,
        ibp_sigma=ibp_sigma
    )

    out_bic = os.path.join(job_dir, "result_bicubic.png")
    out_ours = os.path.join(job_dir, "result_ours.png")
    _save_rgb01(bic_rgb, out_bic)
    _save_rgb01(sr_rgb, out_ours)

    if use_degradation and hr_rgb is not None:
        h_out, w_out, _ = sr_rgb.shape
        hr_rgb_aligned = cv2.resize(hr_rgb, (w_out, h_out), interpolation=cv2.INTER_AREA)

        p_bic = psnr(hr_rgb_aligned, bic_rgb, data_range=1.0)
        s_bic = ssim(hr_rgb_aligned, bic_rgb, data_range=1.0, channel_axis=2)
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



def main():
    import matplotlib.pyplot as plt
    from skimage.metrics import peak_signal_noise_ratio as psnr

    hr_path = "kedi.jpg"  
    scale = 2
    sigma_lr = 1.0

    # method params
    wavelet = "db2"
    beta = 0.8
    thresh = 0.01
    ibp_iters = 10
    ibp_alpha = 0.2
    ibp_sigma = 1.0

    hr_rgb = _to_rgb01(Image.open(hr_path).convert("RGB"))

    # make HR divisible by scale
    H, W, _ = hr_rgb.shape
    H2 = (H // scale) * scale
    W2 = (W // scale) * scale
    hr_rgb = hr_rgb[:H2, :W2, :]
    H, W, _ = hr_rgb.shape

    # simulate LR
    hr_blur = gaussian(hr_rgb, sigma=sigma_lr, channel_axis=-1, preserve_range=True)
    hr_blur = np.clip(hr_blur, 0.0, 1.0)
    lr_rgb = hr_blur[::scale, ::scale, :]

    # bicubic
    lr_pil = Image.fromarray((lr_rgb * 255).astype(np.uint8))
    bic_pil = lr_pil.resize((W, H), resample=Image.BICUBIC)
    bic_rgb = _to_rgb01(bic_pil)

    # wavelet+ibp
    sr_rgb = wavelet_ibp_sr_color(
        lr_rgb,
        scale=scale,
        wavelet=wavelet,
        beta=beta,
        thresh=thresh,
        ibp_iters=ibp_iters,
        ibp_alpha=ibp_alpha,
        ibp_sigma=ibp_sigma
    )

    # PSNR RGB
    psnr_bic = psnr(hr_rgb, bic_rgb, data_range=1.0)
    psnr_sr = psnr(hr_rgb, sr_rgb, data_range=1.0)

    # PSNR Y
    hr_Y, _, _ = rgb_to_ycbcr(hr_rgb)
    bic_Y, _, _ = rgb_to_ycbcr(bic_rgb)
    sr_Y, _, _ = rgb_to_ycbcr(sr_rgb)

    psnr_bic_Y = psnr(hr_Y, bic_Y, data_range=1.0)
    psnr_sr_Y = psnr(hr_Y, sr_Y, data_range=1.0)

    print("=== HR-referenced PSNR (Simulated LR) ===")
    print(f"Bicubic      : RGB {psnr_bic:.2f} dB | Y {psnr_bic_Y:.2f} dB")
    print(f"Wavelet+IBP  : RGB {psnr_sr:.2f} dB | Y {psnr_sr_Y:.2f} dB")

    # show
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 4, 1); plt.imshow(lr_rgb); plt.title("Simulated LR"); plt.axis("off")
    plt.subplot(1, 4, 2); plt.imshow(bic_rgb); plt.title("Bicubic x2"); plt.axis("off")
    plt.subplot(1, 4, 3); plt.imshow(sr_rgb); plt.title("Wavelet+IBP x2"); plt.axis("off")
    plt.subplot(1, 4, 4); plt.imshow(hr_rgb); plt.title("HR (Ref)"); plt.axis("off")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
"""