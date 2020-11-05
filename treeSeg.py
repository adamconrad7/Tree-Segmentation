from tifffile import imread, imwrite
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from scipy.ndimage import gaussian_filter

import skimage.io
import skimage.segmentation as seg
import skimage.color as color
import skimage.feature

import pandas as pd
import numpy as np
from numpy import genfromtxt
import time
import cv2
import math

start_time = time.time()


def downscale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def extract_color(hsv, lower, upper):

    #Find pixels in target range
    mask = cv2.inRange(hsv, lower, upper)
    imask = mask>0
    green = np.zeros_like(hsv, np.uint8)
    green[imask] = hsv[imask]
    rgb = cv2.cvtColor(green, cv2.COLOR_HSV2RGB)
    rgb[~imask] = [255, 255, 255]
    return rgb

def main():

## Path to data
 path = "data/plantation1.tif"

## Percent of data to use
 downscale_percent = 10

## Arbitrary scaling factor for Felzenszwalb segmentation, lower is more accurate and slower
 segment_size_factor = 500

## Read data and downscale for faster proccessing
 rgb0 = imread(path)
 rgb = downscale(rgb0, downscale_percent)
 # rgb0 = downscale(rgb[:int(rgb.shape[0]/3), int(rgb.shape[1]/3):], 20)
 print(rgb.shape)

## Segment image
 image_felzenszwalb = seg.felzenszwalb(rgb, scale=segment_size_factor) ## returns region label of every pixel
 # print(image_felzenszwalb)
 regions = np.unique(image_felzenszwalb, return_index=True, return_inverse=True, return_counts=True)
 # indecies = regions[1]
 # inv_inds = regions[2]
 # print(inv_inds.shape)
 print(f"Felzenszwalb shape: {image_felzenszwalb.shape}")

## Use original image to color the segments based on label
 felz = color.label2rgb(image_felzenszwalb, rgb, kind='avg')

## Display results
 f, axarr = plt.subplots(1,2)
 axarr[0].imshow(rgb)
 axarr[1].imshow(felz)

 print("--- %s seconds ---" % (time.time() - start_time))
 plt.show()

main()

### Other stuff I was playing with:

 # Convert to HSV color space for easier feature detection
 # hsv = cv2.cvtColor(rgb0, cv2.COLOR_RGB2HSV)

## Select color range and return rgb
 # rgb0 = extract_color(hsv, (0, 0, 0), (255, 255, 255))
 # rgb1 = extract_color(hsv, (36, 25, 25), (70, 255, 255))
 # rgb2 = extract_color(hsv, (30, 0, 0), (90, 255, 255))

 # edges = skimage.feature.canny(
 #     image=image,
 #     sigma=2,
 #     low_threshold= 0,
 #     high_threshold=.05,
 # )
 #
 # edges1 = skimage.feature.canny(
 #        image=image,
 #        sigma=2,
 #        low_threshold=.2,
 #        high_threshold=.8,
 # )
 #
 # edges2 = skimage.feature.canny(
 #         image=image,
 #         sigma=2,
 #         low_threshold=.4,
 #         high_threshold=.5,
 # )

## Gaussian filters for texture
 # sig = [1, math.sqrt(2), 2]
 # rfill = gaussian_filter(rgb0[...,0], sigma=20)
 # gfill = gaussian_filter(rgb0[...,1], sigma=20)
 # bfill = gaussian_filter(rgb0[...,2], sigma=20)
 # # rgb0 = np.dstack((rfill,gfill,bfill))
 # filtered0 = gaussian_filter(rgb0, sigma=sig)
 # filtered1 = gaussian_filter(rgb0, sigma=5)
 # filtered2 = gaussian_filter(rgb0, sigma=10)
