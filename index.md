---
layout: default
---

# How to Reverse Engineer Your Camera’s ISP using PyTorch.

## Project in brief
OpenTorchISP is a differentiable camera pipeline designed to automate color grading and reverse-engineer proprietary camera rendering styles.

The Core: A custom PyTorch implementation of a standard ISP pipeline, including Demosaicing, White Balance, Color Correction Matrices (CCM), 1D Look-Up Tables (LUTs), and Lens Shading Correction (LSC).

The Method: Uses gradient descent to optimize these explicit parameters, minimizing the perceptual difference between a RAW input and a reference JPEG.

The Result: A lightweight, interpretable model that can "clone" the look of a specific camera or edit, enabling automated high-fidelity color matching without heavy neural networks.

---

## References
