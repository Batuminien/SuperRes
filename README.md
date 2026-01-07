# CSE464 Digital Image Processing - Super Resolution Project

## Team Members
- **Mustafa Batuhan Değirmenci**
- **Hamide Sıla Akdan**
- **Ebrar Özkan**

---

## Project Overview

This project implements three different **Single Image Super-Resolution (SISR)** methods using classical (non-deep learning) approaches. The goal is to reconstruct high-resolution images from low-resolution inputs while preserving edges, textures, fine details and to compare 3 different approaches achieving this.

---

##  Implemented Methods

### 1. Unified Classical SR (Example-Based + Back-Projection)
**Implementation:** `src/sr_solver.py`

This method combines internal example-based learning with iterative back-projection:
- **Patch Matching:** Searches for similar patches within the image itself at different scales (self-similarity prior)
- **Pyramid Construction:** Builds a Gaussian/Laplacian pyramid to find cross-scale correspondences
- **Back-Projection (IBP):** Iteratively refines the SR result by minimizing reconstruction error between downsampled SR and original LR
- **Optimization:** Solves a linear system balancing classical constraints with example-based texture transfer

### 2. Edge-Based SR with Coherence-Enhancing Shock Filter
**Implementation:** `src/edge_based_sr.py`

A PDE-based approach focusing on edge enhancement:
- **Lanczos Upsampling:** Initial high-quality interpolation
- **Coherence-Enhancing Shock Filter :** Uses directional second derivatives along flow lines to sharpen edges while preserving their natural structure
- **Canny Edge Masking:** Selectively applies sharpening only near detected edges to avoid amplifying noise in flat regions
- **NL-Means Denoising:** Pre-processes the edge detection to reduce false positives
- **Iterative Back-Projection (IBP):** Ensures consistency with the original LR input by minimizing reconstruction error
- **Test-Time Augmentation (Ensemble):** Averages 8 geometric transformations (rotations + flips) for improved quality

### 3. Wavelet + IBP SR
**Implementation:** `src/wavelet_ibp_sr.py`

A frequency-domain approach using wavelet decomposition:
- **Discrete Wavelet Transform (DWT):** Decomposes image into low-frequency (LL) and high-frequency (LH, HL, HH) subbands
- **Subband Upscaling:** Processes each frequency component separately
- **High-Frequency Enhancement:** Boosts detail subbands for sharper results
- **Inverse DWT:** Reconstructs the SR image from enhanced subbands
- **Back-Projection Refinement:** Final consistency enforcement

---

##  Project Structure

```
SuperRes/
├── app.py                 # Flask web application
├── pipeline.py            # SR pipeline orchestration
├── requirements.txt       # Python dependencies
│
├── src/
│   ├── sr_solver.py       # Unified Classical SR implementation
│   ├── edge_based_sr.py   # Edge-Based Shock Filter SR implementation
│   ├── wavelet_ibp_sr.py  # Wavelet + IBP SR implementation
│   ├── patch_matcher.py   # Patch matching utilities
│   ├── pyramid.py         # Image pyramid construction
│   └── utils.py           # Helper functions (I/O, color conversion, etc.)
│
├── templates/
│   ├── index.html         # Upload page
│   └── result.html        # Results display page
│
├── static/
│   ├── uploads/           # Uploaded images
│   ├── results/           # SR output images
│   └── assets/            # Static assets
│
└── input/
    ├── Set5/              # Benchmark dataset
    └── Set14/             # Benchmark dataset
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Installation

1. **Clone or download the repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install flask numpy opencv-python scikit-image scipy pywavelets
   ```

### Running the Web Application

1. **Start the Flask server:**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:5000
   ```

3. **Using the web interface:**
   - Upload an image (PNG, JPG, JPEG, or WebP)
   - Select the SR method (Classical, Edge-Based, or Wavelet+IBP)
   - Choose the upscaling factor (2x, 3x, or 4x)
   - Enable "Use Degradation" for benchmark mode (creates synthetic LR from your image and computes PSNR/SSIM metrics)
   - Click "Run" and view the results

---

##  Method Selection Guide

| Method | Best For | Speed | Quality |
|--------|----------|-------|---------|
| **Classical (Unified)** | Textured images, faces | Slow | High detail preservation |
| **Edge-Based (Shock Filter)** | Images with clear edges | Medium | Sharp edges, natural look |
| **Wavelet + IBP** | General purpose | Fast | Balanced results |

---

##  Notes

- **Degradation Mode:** When enabled, the system artificially downsizes your input image, then upscales it back. This allows computing quality metrics (PSNR/SSIM) against the original.
- **Production Mode:** When degradation is disabled, the input is treated as the actual low-resolution source.
- **Ensemble Mode (Edge-Based):** Improves quality by averaging 8 augmented predictions but takes ~8x longer.

---

## References

- Glasner, D., Bagon, S., & Irani, M. (2009). *Super-resolution from a single image*. ICCV.
- Weickert, J. (2003). *Coherence-enhancing shock filters*. DAGM.
- Mallat, S. (1989). *A theory for multiresolution signal decomposition: the wavelet representation*. IEEE TPAMI.

---

##  License

This project is developed for educational purposes as part of GTÜ-CSE464 Digital Image Processing course.
