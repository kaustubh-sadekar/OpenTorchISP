---
layout: default
---

# How to Reverse Engineer Your Camera’s ISP using PyTorch.

## Project in brief
OpenTorchISP is a differentiable camera pipeline designed to automate color grading and reverse-engineer proprietary camera rendering styles.

The Core: A custom PyTorch implementation of a standard ISP pipeline, including Demosaicing, White Balance, Color Correction Matrices (CCM), 1D Look-Up Tables (LUTs), and Lens Shading Correction (LSC).

The Method: Uses gradient descent to optimize these explicit parameters, minimizing the perceptual difference between a RAW input and a reference JPEG.

The Result: A lightweight, interpretable model that can "clone" the look of a specific camera or edit, enabling automated high-fidelity color matching without heavy neural networks.


## Introduction

Ever taken a photo of a scene that looked beautiful in real life but terrible on your screen? Blown-out skies, crushed shadows, weird tints? That’s because your camera’s ISP (Image Signal Processor) made decisions you didn't ask for. It took the sensor data, cooked it, compressed it, and served you a JPEG you can’t really fix.

But what if we could go back to the source? What if we could build our own ISP in Python to "clone" the look of a pro camera or create our own color science from scratch?


<p align='center'>
  <img src='assets/target_image.png' width="100%">
</p>
<p align='center'>
    A beautiful image captured from Portland's Sellwood bridge (using Google Pixel 7). Your camera sensor does not capture this scene as is. The camera ISP pipeline that ensures your images don't look crappy is even more beautiful that this scene. Trust me!
</p>


### RAW Image

What do we mean by RAW image and how is it different from a JPEG image? A JPEG is like a restaurant meal: cooked, seasoned, and plated. You can’t un-cook the steak or take the salt out of the soup.

A RAW file is the bag of groceries. It’s the raw sensor data—unprocessed, dark, and ugly—but it holds all the potential.

<p align='center'>
  <img src='assets/raw_image_preview.png' width="100%">
</p>
<p align='center'>
    Figure 2 - Raw image captured by the sensor is not coloured. The sensor only records intensity of incident light. Why do we see a checker board pattern as we zoom-in into the raw image?
</p>

```
How do I load and view a raw image?
```

Open source libraries like [rawpy](https://letmaik.github.io/rawpy/api/rawpy.RawPy.html) can be used to read the raw image data from the .dng files. Here is a code snippet you can use:

```python
with rawpy.imread(DNG_PATH) as raw:
    # Bayer mosaic (2D)
    bayer = raw.raw_image_visible.copy().astype(np.float32)
    H, W = bayer.shape

    # Black / white levels
    black = raw.black_level_per_channel
    white = raw.white_level

    # Bayer pattern
    # raw.raw_pattern gives 2x2 mapping of channels: 0=R,1=G,2=B
    pattern = raw.raw_pattern.copy()

    # Camera WB multipliers (if exists)
    cam_wb = raw.camera_whitebalance

print("Raw image info: \nshape: %r,\ndtype: %r,\nmin_val: %r,\nmax_val: %r"%(bayer.shape, bayer.dtype, bayer.min(), bayer.max()))
print("--------------------")
print("black_level_per_channel:", black)
print("white_level:", white)
print("color description:", raw.color_desc)
print("raw_pattern:\n", pattern)
print("camera_whitebalance:", cam_wb)
```
Output
```
Raw image info: 
shape: (2268, 4032),
dtype: dtype('float32'),
min_val: np.float32(1025.0),
max_val: np.float32(16369.0)
--------------------
black_level_per_channel: [1023, 1023, 1023, 1023]
white_level: 16368
color description: b''
raw_pattern:
 [[3 2]
 [0 1]]
camera_whitebalance: [2.2100913524627686, 1.0, 1.454856514930725, 0.0]
```

Sometimes like for this image the `color description` may be an empty string but sometimes it can return the pattern of the color filter array (CFA). In most cases `raw_pattern` returns a 2x2 grid like in this example it returned

Using the CFA pattern we can determine which color filter was used for each pixel and them obtain a colored version of the raw image which is also called mosaic image.

`NOTE: For this example the raw.color_desc returns an empty string so we assume that the CFA pattern is GBRG (left-to-right, top-to-bottom)`

<p align='center'>
  <img src='assets/Mosaiced.png' width="100%">
</p>
<p align='center'>
    Figure 3 - Mosaiced image obtained after extracting the CFA information from the .dng image. Notice the typical green color of the mosaiced raw image because most CFA patterns use twice the pixels to capture green color than red and blue color.
</p>




---

## References
