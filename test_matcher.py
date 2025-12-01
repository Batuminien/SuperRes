import matplotlib.pyplot as plt
import numpy as np
import random
from src.utils import imread_normalized, rgb2yiq
from src.pyramid import create_gaussian_pyramid
from src.patch_matcher import PatchMatcher

def main():
    print("Image is loading and pyramid is preparing...")
    img = imread_normalized("input/Set5/Set5/baby.png")
    img_yiq = rgb2yiq(img)
    y_channel = img_yiq[:, :, 0]
    pyramid = create_gaussian_pyramid(y_channel)
    
    #test scenario
    #lets get a patch from original image (level 0)
    #and lets look at the if there is anything similar into level -2 to that
    source_img = pyramid[0] #query will come from there
    search_img = pyramid[2] #search will be made here (level -2)
    
    print(f"Source Image: {source_img.shape}")
    print(f"Search Image: {search_img.shape}")
    
    #start the matcher (index the search image)
    matcher = PatchMatcher(search_img, patch_size=5)
    
    #choose random patch
    h, w = source_img.shape
    ry = random.randint(0, h - 6)
    rx = random.randint(0, w - 6)
    
    query_patch = source_img[ry:ry+5, rx:rx+5]
    
    #find the match
    print(f"Match is searching...")
    found_patch, found_coord, dist = matcher.find_nearest_neighbors(query_patch)
    
    print(f"Found Location (Level -2): {found_coord}")
    print(f"Similarity Distance: {dist:.4f}")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(query_patch, cmap='gray', vmin=0, vmax=1)
    plt.title("Query Patch\n(Level 0)")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(found_patch, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Found Patch\n(Level -2)\nDist: {dist:.2f}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    
    # show the difference
    # since there may be a DC difference, we equate the averages and look at the difference.
    q_norm = query_patch - np.mean(query_patch)
    f_norm = found_patch - np.mean(found_patch)
    diff = np.abs(q_norm - f_norm)
    
    plt.imshow(diff, cmap='hot')
    plt.title("Texture Difference\n(Black=Perfect)")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()