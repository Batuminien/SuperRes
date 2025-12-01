import numpy as np
from sklearn.neighbors import KDTree

def extract_patches(image, patch_size=5):
    """
    It extracts all possible patchs with size patch_size*patch_size
    
    Args:
        image (2D numpy array): gray level (Y channel) image.
        patch_size (int): Patch size (5x5 used in paper.)
        
    Returns:
        patches: patchs matrix with size (N, patch_size*patch_size)
        coords: (y, x) coordinates with (N, 2) size. 
    """
    
    h, w = image.shape
    patches = []
    coords = []
    
    for y in range(h - patch_size + 1):
        for x in range(w - patch_size + 1):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            patch_vector = patch.flatten()
            
            patches.append(patch_vector)
            coords.append((y, x))
            
    return np.array(patches), np.array(coords)


class PatchMatcher:
    def __init__(self, search_image, patch_size=5):
        """
        It creates an index over the image that will be searched.
        It is generally low resolution layer of the pyramid.
        """
        
        self.search_image = search_image
        self.patch_size = patch_size
        
        #extract patches
        self.patches, self.coords = extract_patches(search_image, patch_size)
        
        #dc removal (extracts average intensity)
        #in the paper they say: "removing their DC (their average grayscale)"
        self.means = np.mean(self.patches, axis=1, keepdims=True)
        self.patches_centered = self.patches - self.means
        
        #create KD-Tree for fast search
        #this operation is made once, after that it is queried many times
        print(f"Creating index... ({len(self.patches)} patch)")
        self.tree = KDTree(self.patches_centered, leaf_size=40)
        
    def find_nearest_neighbors(self, query_patch, k=1):
        """
        It finds most similar patch for given query patch
        """
        
        #extract query patch's DC too
        q_flat = query_patch.flatten().reshape(1, -1)
        q_mean = np.mean(q_flat)
        #search nearest neighbor into tree
        #dist: distance, ind: index of found patch
        q_centered = q_flat - q_mean
        
        dist, ind = self.tree.query(q_centered, k=k)
        
        best_match_idx = ind[0][0]
        match_coord = self.coords[best_match_idx]
        
        found_patch_vector = self.patches[best_match_idx]
        found_patch = found_patch_vector.reshape(self.patch_size, self.patch_size)
        return found_patch, match_coord, dist[0][0]