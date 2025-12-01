import cv2
import numpy as np
from scipy.ndimage import shift
from src.patch_matcher import PatchMatcher, extract_patches
from src.pyramid import create_gaussian_pyramid

def back_projection(high_res, low_res, iterations=20):
    """
    It applies 'Classical SR' restricton in the paper.
    It ensures that the high-resolution estimate is consistent with the low-resolution input.
    """
    
    h_hr, w_hr = high_res.shape
    h_lr, w_lr = low_res.shape
    
    current_hr = high_res.copy()
    
    
    for i in range(iterations):
        #degrade current hr prediction
        #in the paper they use blur+subsample, but here bicubic resize is used
        #blurred = cv2.GaussianBlur(current_hr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        downscaled = cv2.resize(current_hr, (w_lr, h_lr), interpolation=cv2.INTER_CUBIC)
        
        #compute the error
        diff = low_res - downscaled
        
        #enlarge the error and add this into the image (back-project error)
        diff_upscaled = cv2.resize(diff, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        current_hr = current_hr + diff_upscaled
        
    return current_hr

class SRSolver:
    def __init__(self, input_image, scale_factor=1.25):
        self.input_image = input_image
        self.scale_factor = scale_factor
        
        #create the image pyramid (this will be our database)
        self.pyramid = create_gaussian_pyramid(input_image, scale_factor=scale_factor, max_depth=6)
        
        if len(self.pyramid) >= 3:
            self.matcher = PatchMatcher(self.pyramid[2], patch_size=5)
            self.search_level_idx = 2
        else:
            # Resim çok küçükse -1. seviyeyi kullan
            self.matcher = PatchMatcher(self.pyramid[1], patch_size=5)
            self.search_level_idx = 1
            
    """def enhance_details(self, upscaled_img):
        **version1**
        it implements the example-based structure from section 3.3
        it solves the linear constraints of overlapping patches for each pixel.
        
        - find the most similar patch for each pixel according to stride 1.
        - found patch (parent) is a constraint
        - weight this constraint according to its reliability score
        - accumulate and solve all constraints.
        
        
        h, w = upscaled_img.shape
        enhanced_img = upscaled_img.copy()
        patch_size = 5
        
        stride = 1
                
        print(f"Enhancing details...")
        
        
        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                
                # target patch 
                current_patch = upscaled_img[y:y+patch_size, x:x+patch_size]
                
                #find closest patch to this level
                match_patch, (my, mx), dist = self.matcher.find_nearest_neighbors(current_patch)
                
                if dist > 0.1:
                    continue
                    
                #parent yamayı bul
                #eğer yama level -L'de bulunduysa ebeveyni level -(L-1)dedir.
                #örneğin Level -2'de bulduysak Level -1'deki karşılığını alırız.
                parent_level_idx = self.search_level_idx - 1
                parent_img = self.pyramid[parent_level_idx]
                
                #koordinat dönüşümü (Low -> High)
                py = int(my * self.scale_factor)
                px = int(mx * self.scale_factor)
                
                #sınır kontrolü
                if py + patch_size > parent_img.shape[0] or px + patch_size > parent_img.shape[1]:
                    continue
                    
                parent_patch = parent_img[py:py+patch_size, px:px+patch_size]
                
                cy, cx = y + patch_size//2, x + patch_size//2
                #kısıtlamayı ekle
                #high freq transfer: Current + (Parent - Match)
                
                high_freq_detail = parent_patch - match_patch
                enhanced_img[y:y+patch_size, x:x+patch_size] += high_freq_detail * 0.5
                        
        return enhanced_img
        """
        
    def enhance_details(self, upscaled_img):
        h, w = upscaled_img.shape
        patch_size = 5
        
        reconstruction = np.zeros_like(upscaled_img, dtype=np.float32)
        weight_map = np.zeros_like(upscaled_img, dtype=np.float32)
        
        stride = 1
        
        total_patches = 0
        passed_patches = 0
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                
                total_patches += 1
                
                current_patch = upscaled_img[y:y+patch_size, x:x+patch_size]
                
                match_patch, (my, mx), dist = self.matcher.find_nearest_neighbors(current_patch)
                
                parent_level_idx = self.search_level_idx - 1
                parent_img = self.pyramid[parent_level_idx]
                
                py = int(my * self.scale_factor)
                px = int(mx * self.scale_factor)
                
                if py + patch_size > parent_img.shape[0] or px + patch_size > parent_img.shape[1]:
                    continue
                
                high_res_patch = parent_img[py:py+patch_size, px:px+patch_size]
                
                if dist > 0.05:
                    continue
                
                passed_patches += 1
                
                weight = np.exp(-dist * 10.0)
                
                if weight < 0.001:
                    continue
                
                detail_patch = current_patch + (high_res_patch - match_patch)
                
                reconstruction[y:y+patch_size, x:x+patch_size] += detail_patch * weight
                weight_map[y:y+patch_size, x:x+patch_size] += weight
        
        ratio = (passed_patches / total_patches) * 100 if total_patches > 0 else 0
        print(f"  -> Patch Stats: {passed_patches}/{total_patches} passed (Ratio: {ratio:.2f}%)")
        
        mask = weight_map > 0.0001
        
        final_img = upscaled_img.copy()
        final_img[mask] = reconstruction[mask] / weight_map[mask]
        
        return final_img
    
    def upscale(self, target_scale=2.0):
        """
        main SR loop.
        """
        
        current_img = self.input_image.copy()
        current_scale = 1.0
        
        step_count = 0
        while current_scale < target_scale:
            step_count += 1
            next_scale = current_scale * self.scale_factor
            if next_scale > target_scale:
                next_scale = target_scale
                
            print(f"\n--- Step {step_count}: Scale x{next_scale:.2f} ---")
            
            #bicubic upscale (first simple upscale)
            h, w = current_img.shape
            new_h = int(h * (next_scale / current_scale))
            new_w = int(w * (next_scale / current_scale))
            upscaled_img = cv2.resize(current_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            #example based enchancement (glasner method)
            #this step sharpened the texture
            enhanced_img = self.enhance_details(upscaled_img)
            
            #back-projection (error correction)
            #this step provides stay the image consistent.
            final_img = back_projection(enhanced_img, self.input_image)
            
            current_img = final_img
            current_scale = next_scale
            
        return current_img