from tifffile import imread, imwrite
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from scipy.ndimage import gaussian_filter, label, find_objects
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
import random

from tensorflow import keras


from sklearn.cluster import KMeans
from skimage.color import rgb2hsv
import os
from os import listdir

model = keras.models.load_model('model/')


start_time = time.time()

def chunk_from_coords(im, x, y, dim):
    return im[x:x+dim, y:y+dim]

def get_data(tiles, n):
    arr =[]
    i = 0


    while i < n:
        idx = random.randint(0, len(tiles))

        # print(tiles[idx].sum(axis = 0))
        if tiles[idx].sum() != 0:
            arr.append(tiles[idx])
            i += 1


            # continue
    return np.array(arr)

def chunkify(img, dim):
    chunks = []
    print("image shape before chunking: ", img.shape)
    xmax = img.shape[0] - (img.shape[0] % dim)
    ymax = img.shape[1] - (img.shape[1] % dim)

    tiles = [img[x:x+dim,y:y+dim]for x in range(0,xmax,dim) for y in range(0,ymax,dim)]
    # print(type(tiles))
    # print(tiles[0])
    tiles = np.array(tiles)
    # print("image shape after chunking: "img.shape)

    # print(tiles.shape)
    # print(len(tiles)  )
    # print(len(tiles) % 16 )
    # print(len(tiles) - len(tiles) % 16 )
    # print((len(tiles) - len(tiles) % 16) % 16 )

    print("shape in chunkify: ", tiles.shape)
    print("element shape in chunkify: ", tiles[0].shape)



    if (len(tiles) % 16) != 0:
        print("DELETE CALLED!")
        tiles = np.delete(tiles, np.s_[-(len(tiles) % 16):], axis=0)

    # print(len(tiles))
    # print(tiles[0])
    # print(len(tiles)  )

    # print( len(tiles) % 16)
    reshaped =np.reshape(tiles,(-1, dim, dim, 4))
    # print(reshaped[0])
    return reshaped

def downscale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # print("dim: ", width, height)

    r = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return r

def extract_color(hsv, lower, upper):

    #Find pixels in target range
    mask = cv2.inRange(hsv, lower, upper)
    imask = mask>0
    green = np.zeros_like(hsv, np.uint8)
    green[imask] = hsv[imask]
    rgb = cv2.cvtColor(green, cv2.COLOR_HSV2RGB)
    rgb[~imask] = [255, 255, 255]
    return rgb

def k_means(im):
    n = im.shape[0]
    m = im.shape[1]
    z = np.dstack((im,rgb2hsv(im)))
    vectorized = np.float32(z.reshape((-1,6)))
    kmeans = KMeans(random_state=0, init='random', n_clusters=6)
    labels = kmeans.fit_predict(vectorized)
    lookupTable, idx,  labels, counts = np.unique(labels, return_inverse=True, return_counts=True, return_index=True)
    print("Lookup table: ",lookupTable, '\n')
    print("Indecies: ",idx, '\n')
    print("labels: ",labels, '\n')
    print("counts: ",counts, '\n')
    print("n pixels: ", im.shape[0]*im.shape[1])
 # print("n regions?: ", len(labels))
 # print("n regions?: ", len(idx))
    print("n regions: ", len(counts))
    pic = labels.reshape(n,m)
    # pic2 = color.label2rgb(pic, im, kind='overlay')
    # return
    return pic

def read_segs(path):
    ims = []
    labels = []
    for classname in listdir(path):
        classpath = os.path.join(path, classname)
        for i, file in enumerate(listdir(classpath)):
            # if(i == 16):
                # continue
            # if i > 163:
                # break
            seg = imread(os.path.join(classpath, file))
            ims.append(seg)
            labels.append(classname)

    tup = [np.array(ims), np.array(labels)]
    return tup

def main():

## Path to data
 # path = "data/plantation1.tif"
 path = "data/Mac_1120_medium_012021.tif"


## Percent of data to use
 downscale_percent = 100
## Arbitrary scaling factor for Felzenszwalb segmentation, lower is more accurate and slower
 segment_size_factor = 20

## Read data and downscale for faster proccessing
 rgb0 = imread(path)
 rgb0 = (rgb0/256).astype('uint8') #for plantation 3 color depth
 # rgb = downscale(rgb0[:int(rgb0.shape[0]/3), int(rgb0.shape[1]/3):], downscale_percent)
 rgb = downscale(rgb0, downscale_percent)

 # segs = read_segs('big_chunks/sapling')
 # segs = segs[0]
 #

 allsegs = []
 classpath = 'big_chunks/sapling'
 # for file in listdir('big_chunks/sapling'):
 #     seg = imread(os.path.join(classpath, file))
 #     allsegs.append(seg)

     # print(file)
 # allsegs = np.array(allsegs)
 # print(allsegs.shape)

 # print(allsegs[0].shape)
 # segs = allsegs
 # segs.append(allsegs[30])
 #
 # segs.append(allsegs[31])
 # segs.append(allsegs[32])
 # segs.append(allsegs[33])
 # segs.append(allsegs[34])
 #
 #
 # segs.append(allsegs[37])

 # print(len(segs))

 # segs.append()
 # ncols = 10
 #
 # fig, axs = plt.subplots(int(len(segs) /  ncols), ncols)
 # axs = axs.flatten()
 # for img, ax in zip(segs, axs):
 #      ax.imshow(img)
 #      ax.set_xticks([])
 #      ax.set_yticks([])

 fig, axs = plt.subplots()
 axs.imshow(rgb)
 plt.show()

 # tiles = chunkify(rgb, 2)
 # segs = get_data(tiles, 49)
 # print(segs[0])

 # segs = []
 # segs.append(chunk_from_coords(rgb0, 5140, 390, 2))
 # segs.append(chunk_from_coords(rgb0, 390, 5140, 2))
 # segs = np.array(segs)
 #
 # print(segs)
 #
 # y_pred = model.predict_classes(segs)
 # print(y_pred)
 # labels = ['bark', 'earth', 'grass', 'road', 'sapling']
 #
 #
 # f, axes = plt.subplots(1, 2)
 # for i, ax in enumerate(axes.ravel()):
 #     ax.imshow(segs[i])
 #     ax.title.set_text(labels[y_pred[i]])
 #     ax.axes.xaxis.set_ticks([])
 #     ax.axes.yaxis.set_ticks([])
 # plt.show()

#    0       1       2      3       4

 # seg = k_means(rgb[...,:-1])
 # labels1 = seg.slic(rgb[...,:-1], compactness=1, n_segments=6,  convert2lab=True, enforce_connectivity=False)
 # lookupTable, idx,  labels, counts = np.unique(labels1, return_inverse=True, return_counts=True, return_index=True)
 # print("Lookup table: ",lookupTable, '\n')
 # print("Indecies: ",idx, '\n')
 # print("labels: ",labels, '\n')
 # print("counts: ",counts, '\n')
 # print("n pixels: ", rgb.shape[0]*rgb.shape[1])
 # # print("n regions?: ", len(labels))
 # # print("n regions?: ", len(idx))
 # print("n regions: ", len(counts))
 # print(labels1.shape)
 #
 #
 # mask = np.zeros(rgb.shape[:2], dtype = "uint8")
 # mask[labels1 == 0] = 255
 # f, axarr = plt.subplots(1,1)
 # axarr.imshow(cv2.bitwise_and(rgb, rgb, mask = mask))
 # plt.show()
 # for (i, segVal) in enumerate(np.unique(labels1)):
 #    mask = np.zeros(rgb.shape[:2], dtype = "uint8")
 #    mask[labels1 == segVal] = 255
 #    axarr[i].imshow(cv2.bitwise_and(rgb, rgb, mask = mask))
 #


 # slic= color.label2rgb(labels1, rgb, kind='avg')
 # slic= labels1
 #
 # slices = find_objects(slic)
 # print(objs)

 # for (i, segVal) in enumerate(np.unique(labels1)):
	# # construct a mask for the segment
 #        # print "[x] inspecting segment %d" % (i)
    	# mask = np.zeros(rgb.shape[:2], dtype = "uint8")
    	# mask[labels1 == segVal] = 255
 #    	# show the masked region
 #    	cv2.imshow("Mask", mask)
 #    	cv2.imshow("Applied", cv2.bitwise_and(rgb, rgb, mask = mask))
 #    	cv2.waitKey(0)

 # numobjects = 60000
 #
 # fig, axes = plt.subplots(ncols=numobjects)
 # for ax, sli in zip(axes.flat, slices):
 #        ax.imshow(labels1[sli], vmin=0, vmax=numobjects)
 #        tpl = 'BBox:\nymin:{0.start}, ymax:{0.stop}\nxmin:{1.start}, xmax:{1.stop}'
 #        ax.set_title(tpl.format(*sli))
 # fig.suptitle('Individual Objects')
 #
 # plt.show()
 # print(objs.shape)

 # adj2 = np.ones((3,3), dtype='bool')
 #
 # adj3 = np.ones((3,3,3), dtype='bool')
 #
 # labels, nfeatures = label(slic, adj2)
 # labels2, nfeatures = label(rgb[...,:-1], adj3)
 #
 # print(nfeatures)




 # means = k_means(rgb[...,:-1])

 # seg[...,:-1] = 255

## Segment image
#  image_felzenszwalb = seg.felzenszwalb(rgb, scale=segment_size_factor) ## returns region label of every pixel
#  # print(image_felzenszwalb)
 # lookupTable, idx,  labels, counts = np.unique(image_felzenszwalb, return_inverse=True, return_counts=True, return_index=True)
 # # print("Lookup table: ",lookupTable, '\n')
 # # print("Indecies: ",idx, '\n')
 # # print("labels: ",labels, '\n')
 # # print("counts: ",counts, '\n')
 # print("n pixels: ", rgb.shape[0]*rgb.shape[1])
 # # print("n regions?: ", len(labels))
 # # print("n regions?: ", len(idx))
 # print("n regions: ", len(counts))
 #
#  # print("n distinct labels: ", len(lookupTable))
#
#
#  # inv_inds = regions[2]
#  # print(inv_inds.shape)
# #  print(f"Felzenszwalb shape: {image_felzenszwalb.shape}")
# #
# # ## Use original image to color the segments based on label
 # felz = color.label2rgb(image_felzenszwalb, rgb, kind='avg')

 # print
 # edges = skimage.feature.canny(
 #      image=rgb,
 #      sigma=2,
 #      low_threshold= 0,
 #      high_threshold=.05,
 # )

## Display results
 # f, axarr = plt.subplots(1,3)
 # axarr[0].imshow(rgb)
 # axarr[1].imshow(labels1)
 # axarr[2].imshow(cv2.bitwise_and(rgb, rgb, mask = mask))

 #
 print("--- %s seconds ---" % (time.time() - start_time))
 # plt.show()

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
