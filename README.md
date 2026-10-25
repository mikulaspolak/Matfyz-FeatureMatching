# Matfyz-FeatureMatching

An academic computer-vision experiment that compares local feature matching methods on pairs of images.

## What it does

The script loads two grayscale images and saves visualizations of their correspondences using:

- ORB descriptors with brute-force Hamming matching.
- SIFT descriptors with brute-force matching and Lowe's ratio test.
- SIFT descriptors with FLANN matching and Lowe's ratio test.

This project focuses on finding and visualizing image correspondences. It does not estimate camera geometry or reconstruct a 3D point cloud.

## Input images

The project contains image-pair folders such as `izba`, `krakow`, and `washington`. Each selected folder is expected to contain:

```text
img1.png
img2.png
```

The folder is currently selected by the `priecinok` variable near the top of `FeatureMatching.py` (initially `izba`). Set it to the folder you want to process.

## Requirements

The project was developed with an older Python/OpenCV environment. The original script recommends the OpenCV contrib package because it calls `cv.xfeatures2d.SIFT_create`:

```bash
python -m pip install opencv-contrib-python==3.4.2.17 matplotlib
```

That OpenCV version is old and may require a compatible Python version. The checked-in virtual environment is machine-specific and should not be used as a portable dependency specification.

## Run

From this directory, after selecting an input folder and installing compatible dependencies:

```bash
python FeatureMatching.py
```

The script runs all three methods when started and writes the resulting match visualizations into the selected image folder.

## Academic context and relationship to 3D reconstruction

This is remembered as diploma-thesis-related academic work; confirm its title and year before publishing.

It is a separate codebase from the 3D reconstruction project. The two share the topic of image features and matching, but this script ends at visualizing correspondences while the reconstruction project uses feature matches as one step in a larger camera-geometry and triangulation pipeline. The code alone does not establish whether they were separate thesis deliverables or parts of the same thesis.

## Limitations and attribution

- The selected folder is hard-coded, and the script executes immediately rather than exposing a command-line interface.
- The source references the OpenCV feature-matching tutorial. Retain appropriate attribution if adapting that material.
- Check the rights to the included image pairs before making them public.
