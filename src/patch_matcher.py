import numpy as np
from sklearn.neighbors import KDTree

def extract_patches(image, patch_size=5):
    """
    Extracts all possible overlapping patches from a given 2D image using a sliding window approach.
    
    Args:
        image (np.ndarray): Input 2D grayscale image (Y channel) with shape (H, W).
        patch_size (int, optional): The height and width of the square patches. Defaults to 5.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - patches: A matrix of flattened patches with shape (N, patch_size*patch_size).
            - coords: An array of (y, x) coordinates representing the top-left corner of each patch with shape (N, 2).
    """
    
    h, w = image.shape
    patches = []
    coords = []
    
    # Iterate over the image with a sliding window
    for y in range(h - patch_size + 1):
        for x in range(w - patch_size + 1):
            # Extract the patch region
            patch = image[y:y+patch_size, x:x+patch_size]
            
            #Flatten the 2D patch to a 1D vector for feature representation
            patch_vector = patch.flatten()
            
            patches.append(patch_vector)
            coords.append((y, x))
            
    return np.array(patches), np.array(coords)


class PatchMatcher:
    """
    A class to facilitate fast nearest-neighbor searching of image patches.
    
    This class builds a KD-Tree index on the patches of a search image to allow
    efficient querying. It includes preprocessing steps like DC removal (mean substraction)
    to focus on structural texture rather than absolute intensity.
    """
    
    def __init__(self, search_image, patch_size=5):
        """
        Initializes the PatchMatcher by building an index over the provided image.
        
        Args:
            search_image (np.ndarray): The source image to search within.
            patch_size (int, optional): The size of the patches. Defaults to 5.
        """
        
        self.search_image = search_image
        self.patch_size = patch_size
        
        # Extract all patches and their coordinates from the search image
        self.patches, self.coords = extract_patches(search_image, patch_size)
        
        # DC Removal (Mean Substraction)
        # As described in SR literature, removing the averaga intensity
        # ensures that matching is invariant to local brightness changes and focuses on texture.
        self.means = np.mean(self.patches, axis=1, keepdims=True)
        self.patches_centered = self.patches - self.means
        
        # Build KD-Tree for efficient nearest neighbor search.
        # This is a one-time computationally expensive operation that speeds up subsequent queries.
        print(f"Creating index... ({len(self.patches)} patch)")
        self.tree = KDTree(self.patches_centered, leaf_size=40)
        
    def find_nearest_neighbors(self, query_patch, k=1):
        """
        Finds the most similar patch in the search image to the given query patch.
        
        Args:
            query_patch (np.ndarray): The input patch to find a match for. Shape should be (patch_size, patch_size).
            k (int, optional): The number of nearest neighbors to find. Defaults to 1.
            
        Returns:
            Tuple[np.ndarray, tuple, float]:
                - found_patch: The best matching patch reconstructed from the search image.
                - match_coord: The (y, x) coordinates of the matching patch in the search image.
                - distance: The Euclidean distance between the query and the match (in feature space).
            """
        
        # Flatten the query patch to match the indexed data format
        q_flat = query_patch.flatten().reshape(1, -1)
        
        # Remove DC component from the query patch to match the indexing strategy
        q_mean = np.mean(q_flat)
        q_centered = q_flat - q_mean
        
        # Query the KD-Tree
        # dist: Distances to the nearest neighbors
        # ind: Indices of the nearest neighbors in the self.patches array
        dist, ind = self.tree.query(q_centered, k=k)
        
        # Extract the best match (k=1 result)
        best_match_idx = ind[0][0]
        match_coord = self.coords[best_match_idx]
        
        #Retrieve the original pixel values of the matched patch
        found_patch_vector = self.patches[best_match_idx]
        found_patch = found_patch_vector.reshape(self.patch_size, self.patch_size)
        return found_patch, match_coord, dist[0][0]