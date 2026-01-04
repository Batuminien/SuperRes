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
    Performs Iterative Back-Projection to enforce consistency between the
    reconstructed HR image and the original LR image.
    
    The reconstruction constraint states that downsampling the SR result should yield
    the original LR image:
        LR ≈ (HR * Gaussian) ↓ s
        
    Args:
        high_res (np.ndarray): Initial high-resolution estimate (2D array).
        low_res (np.ndarray): Original low-resolution image (2D array).
        downscale_factor (int, optional): The downscaling factor 's'. Defaults to 2. (e.g, 2, 3, or 4)
        sigma (float, optional): Standard deviation for Gaussian blur. If None, computed from downscale_factor. Defaults to None.
        iterations (int, optional): Number of back-projection iterations. Defaults to 10.
        alpha (float, optional): Step size for the update. Defaults to 0.2.
        
    Returns:
        np.ndarray: The refined high-resolution image after back-projection.
    """

    hr = high_res.astype(np.float32)
    lr = low_res.astype(np.float32)

    h_hr, w_hr = hr.shape
    h_lr, w_lr = lr.shape

    # Calculate PSF sigma if not provided (Standard formulation in SR literature)
    if sigma is None:
        sigma = np.sqrt(downscale_factor ** 2 - 1.0)

    current_hr = hr.copy()

    for _ in range(iterations):
        # Simulate the imaging process (Blur + Downsample)
        if sigma > 1e-6:
            blurred = cv2.GaussianBlur(
                current_hr, (0, 0),
                sigmaX=sigma, sigmaY=sigma
            )
        else:
            blurred = current_hr

        # Handle downsampling (check for integer scaling)
        if (h_hr == h_lr * downscale_factor) and (w_hr == w_lr * downscale_factor):
            # Perfect integer scale: use strided slicing
            current_lr = blurred[::int(downscale_factor), ::int(downscale_factor)]
        else:
            # Non-integer scale: use area interpolation
            current_lr = cv2.resize(
                blurred,
                (w_lr, h_lr),
                interpolation=cv2.INTER_AREA
            )

        # Calculate reconstruction error in LR domain.
        diff = lr - current_lr

        # Back-project error to HR domain
        up = cv2.resize(
            diff,
            (w_hr, h_hr),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Update HR estimate
        # Limit the correction to prevent ringing artifacts or instability.
        correction = alpha*up
        limit = 0.03
        correction = np.clip(correction, -limit, limit)

        current_hr = current_hr + correction
        
    return np.clip(current_hr, 0.0, 1.0)


class SRSolver:
    """
    Unified Single Image SR Solver.
    
    This class combines:
    1. Example-Based SR: Uses patch recurrence across scales (from an internal image pyramid)
                         to enhance high-frequency details.
    2. Classical SR: Uses back-projection to ensure the result is consistent with the input.
    """
    
    def __init__(self, input_image, scale_factor=1.25):
        """
        Args:
            input_image (np.ndarray): The low-resolution input image (Y channel).
            scale_factor (float, optional): The scale factor for the image pyramid. Defaults to 1.25.
        """
        self.input_image = input_image
        self.scale_factor = scale_factor
        
        # Create gaussian pyramid to serve as the internal patch database
        # We look for similar patches in these lower-resolution versions of the image.
        self.pyramid = create_gaussian_pyramid(self.input_image, scale_factor=scale_factor, max_depth=6)
        
        # Initialize PatchMatchers for each pyramid level for fast NN search
        self.matchers = []
        for lvl in range(1, len(self.pyramid)):
            img_lvl = self.pyramid[lvl]
            matcher = PatchMatcher(img_lvl, patch_size=5)
            self.matchers.append(matcher)
        
        # Keep track of which pyramid level corresponds to which matcher
        self.matcher_level_indices = list(range(1, len(self.pyramid)))    
    
    def enhance_details_ls(self, upscaled_img):
        """
        Calculates the example-based high frequency details using Least Squares formulation.
        
        For each patch in the upscaled_img:
            1) Finds similar patches in the lower levels of the pyramid.
            2) Learns the high-frequency detail (Parent - Child difference).
            3) Transfers this detail to the current image.
            
        Args:
            upscaled_img (np.ndarray): The current intermediate HR estimate.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - num_ex: Numerator accumulator for the LS equation.
                - den_ex: Denominator (weight) accumulator for the LS equation.
        """
        
        h, w = upscaled_img.shape
        patch_size = 5
        stride = 1
        
        num_ex = np.zeros_like(upscaled_img, dtype=np.float32)
        den_ex = np.zeros_like(upscaled_img, dtype=np.float32)
        
        total_patches = 0
        used_patches = 0
        
        print(f"DEBUG: Max pixel value: {upscaled_img.max():.4f}")
        
        # Pre-calculation for adaptive thresholding
        # We do not want to waste time matching flat/smooth patches (like sky or walls).
        all_patches, _ = extract_patches(upscaled_img, patch_size)
        all_stds = np.std(all_patches, axis=1)
        
        # Dynamic variance threshold
        # Ignore patches smoother than the 90th percentile noise floor
        variance_threshold = np.percentile(all_stds, 90)
        variance_threshold = max(variance_threshold, 0.005)
        print(f"Dynamic Variance Threshold: {variance_threshold:.5f}")
        
        # Estimate global matching quality (median distance) to normalize weights
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
        
        # Main Loop: Patch Matching and Detail Transfer
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):

                total_patches += 1

                current_patch = upscaled_img[y:y+patch_size, x:x+patch_size]

                # Skip flat patches to speed up processing
                if np.std(current_patch) < variance_threshold:
                    continue

                # Accumulators for multi-scale details for this specific patch position
                multi_detail_num = np.zeros((patch_size, patch_size), dtype=np.float32)
                multi_detail_den = 0.0

                # Search in all available pyramid levels
                for idx, matcher in enumerate(self.matchers):
                    level      = self.matcher_level_indices[idx]
                    coarse_img = self.pyramid[level]                # The database image (LR)
                    parent_img = self.pyramid[level - 1]            # The detailed image (HR relative to coarse)

                    # Calculate local noise floor (self-distance)
                    patch_centered = current_patch - current_patch.mean()
                    shifted_patch  = np.roll(patch_centered, shift=1, axis=0)
                    self_dist      = np.linalg.norm(patch_centered - shifted_patch)
                    self_dist      = max(self_dist, 1e-6)

                    # Find nearest neigbor
                    match_patch, (my, mx), dist = matcher.find_nearest_neighbors(current_patch)

                    # Filter poor matches
                    dist_threshold = min(0.7 * self_dist, 1.2 * median_dist)
                    if dist > dist_threshold:
                        continue

                    # Retrieve the corresponding high-frequency detail from the parent level
                    # Coordinate mapping: Coarse to Fine
                    py = int(my * self.scale_factor)
                    px = int(mx * self.scale_factor)
                    
                    # Boundary check
                    if (py + patch_size > parent_img.shape[0] or
                        px + patch_size > parent_img.shape[1]):
                        continue

                    parent_patch = parent_img[py:py+patch_size, px:px+patch_size]

                    # Calculate weight based on similarity
                    normalized_dist = dist / median_dist if median_dist > 0 else dist
                    beta   = 1.5
                    weight = np.exp(-beta * normalized_dist)
                    if weight < 0.15:
                        continue

                    # Extract detail: HR_patch - LR_patch
                    detail = parent_patch - match_patch

                    multi_detail_num += weight * detail
                    multi_detail_den += weight

                if multi_detail_den <= 0:
                    continue

                # Average the learned details
                mean_detail = multi_detail_num / multi_detail_den

                # Construct the target patch: Original + Scaled Detail
                # detail_scale can be adjusted to control sharpness intensity
                detail_scale = 0.25
                target_patch = current_patch + detail_scale * mean_detail

                # Add to global least squares accumulators
                patch_weight = multi_detail_den
                num_ex[y:y+patch_size, x:x+patch_size] += patch_weight * target_patch
                den_ex[y:y+patch_size, x:x+patch_size] += patch_weight

                used_patches += 1

        ratio = (used_patches / total_patches) * 100 if total_patches > 0 else 0
        print(f"  -> Patch Stats: {used_patches}/{total_patches} used (Ratio: {ratio:.2f}%)")

        return num_ex, den_ex
    
    def upscale(self, target_scale=2.0,
                lambda_ex=1.0, lambda_cl=1.0, use_degradation=True):
        """
        Executes the Coarse-to-Fine Super-Resolution pipeline.
        
        Instead of upscaling directly to x4 (which is hard), we perform small steps
        (e.g., x1.25 -> x1.56 -> ... -> x4).
        
        In each step:
          1. Bicubic Interpolation (Initial guess)
          2. Example-Based Enhancement (Hallucinate details from pyramid)
          3. Back-Projection (Enforce consistency with original input)
          4. Merge results using weighted Least Squares.

        Args:
            target_scale (float): Final desired scale (e.g., 2.0, 4.0).
            lambda_ex (float): Weight for Example-Based term.
            lambda_cl (float): Weight for Classical Back-Projection term.
            use_degradation (bool): Whether to apply strict reconstruction constraints 
                                    (set False for real-world images without GT).

        Returns:
            np.ndarray: The final Super-Resolved image.
        """

        lr = self.input_image.astype(np.float32)
        h_lr, w_lr = lr.shape

        # Start from the original scale
        current_hr = lr.copy()
        current_scale = 1.0
        step_id = 0
        
        downscale_factor = int(round(target_scale))
        bp_sigma = None

        # Coarse-to-Fine loop
        while current_scale < target_scale - 1e-6:
            step_id += 1

            # Determine the scale for this step so we can ensure we do not exceed target_scale
            remaining = target_scale / current_scale
            step_factor = min(self.scale_factor, remaining)
            new_scale = current_scale * step_factor

            h_hr = int(round(h_lr * new_scale))
            w_hr = int(round(w_lr * new_scale))
            
            is_last_step = (abs(new_scale - target_scale) < 1e-6)

            print(f"\n=== Coarse-to-fine Step {step_id}: "
                  f"scale {current_scale:.3f} -> {new_scale:.3f} "
                  f"({h_hr}x{w_hr}) ===")

            # Bicubic upscale
            bicubic = cv2.resize(
                current_hr,
                (w_hr, h_hr),
                interpolation=cv2.INTER_CUBIC
            )
            bicubic = np.clip(bicubic, 0.0, 1.0)

            # Example-based Enhancement
            print("  -> Example-based LS")
            num_ex, den_ex = self.enhance_details_ls(bicubic)

            # Classical Back-Projection
            # Adjust iterations based on wheter it is the final step or an intermediate one
            if use_degradation:
                if is_last_step:
                    print("  -> Classical SR back-projection (LAST STEP)")
                    bp_iters=10
                else:
                    print("  -> Classical SR back-projection (INTERMEDIATE STEP)")
                    bp_iters = 15
            else:
                # For real-world images (no degradation), we use fewer iterations to avoid artifacts
                if is_last_step:
                    print("  -> Classical SR back-projection (LAST STEP)")
                    bp_iters=2
                else:
                    print("  -> Classical SR back-projection (INTERMEDIATE STEP)")
                    bp_iters = 5
                
            hr_bp = back_projection(
                bicubic,
                lr,
                downscale_factor=downscale_factor, # Always enforce consistency with original LR
                sigma=bp_sigma,
                iterations=bp_iters,
                alpha=0.2,
            )
            hr_bp = np.clip(hr_bp, 0.0, 1.0)
            
            # Merge results (Weighted Least Squares Combination)
            # Formula: (lambda_ex * Ex_Term + lambda_cl * BP_Term) / (lambda_ex * Ex_Weight + lambda_cl)
            lam_ex_step = lambda_ex
            lam_cl_step = lambda_cl
            
            num_total = lam_ex_step * num_ex + lam_cl_step * hr_bp
            den_total = lam_ex_step * den_ex + lam_cl_step
            
            updated = hr_bp.copy()
            mask = den_total > 0.0
            updated[mask] = num_total[mask] / den_total[mask]
            updated = np.clip(updated, 0.0, 1.0)

            current_hr = updated
            current_scale = new_scale

        return current_hr
