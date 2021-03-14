import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from os import listdir
from os.path import isfile, join
import skimage.segmentation as seg
import skimage.color as color
import skimage.future.graph as graph
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
# import osgeo
# from osgeo import gdal
# import gdal
import cv2

import numpy as np
import os
import time
import math

import sys



start_time = time.time()

# ncoords = []
# def onclick(event):
#     global ix, iy
#     ix, iy = event.xdata, event.ydata
#     print(ix, iy)
#
#     global ncoords
#     ncoords.append((ix, iy))
def onclick(event):
    if event.dblclick:
        print("(",int(event.xdata),",",int(event.ydata),"),")



    # if len(ncoords) == 2:
    #     f.canvas.mpl_disconnect(cid)

    # return ncoords

def read_coords(path):
    segs = []
    for classname in listdir(path):
        classpath = os.path.join(path, classname)
        for file in listdir(classpath):
            coords = file.split(".")[0]
            seg = imread(os.path.join(classpath, file))
            tup = [int(coords.split(",")[0]), int(coords.split(",")[1])]
            segs.append(tup)

    return segs

def crop(img, x, y, dim):
    xstart = int(x - (dim/2))
    ystart = int(y - (dim/2))

    xend = int(x + (dim/2))
    yend = int(y + (dim/2))

    return img[ystart:yend, xstart:xend]

def crop_and_write(img, x, y, dim, dest):
    new = img[y:y+dim, x:x+dim]
    imwrite(dest + str(x) + ',' + str(y) + '.tif', new)
    return new

def crop_and_write_from_mid(img, x, y, dim, dest):
    xstart = int(x - (dim/2))
    xend = int(x + (dim/2))
    ystart = int(y - (dim/2))
    yend = int(y + (dim/2))

    # new = img[y:y+dim, x:x+dim]
    new = img[ystart:yend, xstart:xend]
    imwrite(dest + str(x) + ',' + str(y) + '.tif', new)
    return new

def find_xy(im):
    f, axes = plt.subplots(1,1)
    axes.imshow(im)
    plt.show()

def verify_chunks(segs):
    nrows = math.sqrt(len(segs))
    if nrows % 1 != 0:
        nrows = nrows - nrows % 1
    ncols = len(segs)/nrows


    fig, axs = plt.subplots( int(nrows), int(ncols))
    axs = axs.flatten()
    for i, (ax, seg) in enumerate(zip(axs, segs)):
        ax.imshow(seg)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(i)

    # for i, (label, ax) in enumerate(zip(lookupTable, ax)):
    #     mask = np.zeros(im.shape[:2], dtype = "uint8")
    #     mask[labels == label] = 255
    #     ax.imshow(cv2.bitwise_and(im, im, mask = mask))

    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()

    # f, axes = plt.subplots(7,8)
    # for i, ax in enumerate(axes.ravel()):
    #     ax.imshow(segs[i])
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        #
    # plt.show()

def read_segs(path):
    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # for file in listdir('training/bark/'):
    #     print(file)
    # print(listdir(path))
    segs = []
    for classname in listdir(path):
        classpath = os.path.join(path, classname)
        for file in listdir(classpath):
            print(file)
            seg = imread(os.path.join(classpath, file))
            tup = [seg, classname]
            segs.append(tup)

    return np.array(segs)
            # print(file)

def mark_img(im, coords):
    dim = 30
    for coord in coords:
        xstart = int(coord[0] - (dim/2))
        ystart = int((coord[1]) - (dim/2))
        xend = int(coord[0] + (dim/2))
        yend = int( (coord[1]) + (dim/2))

        mark = np.full((4), [255, 0, 0, 255])
        im[ystart:yend, xstart:xend] = mark
    return im



        # if not listdir(classpath):
        #     print("{} is empty", classpath)
        # else:
        #     for file in listdir(classpath):
        #         print(file)

    # print(onlyfiles)

def chunk_from_coords(path, im, dim):
    segs = []
    for classname in listdir(path):
        classpath = os.path.join(path, classname)
        for file in listdir(classpath):
            # if file.find('sapling'):
            if 'sapling' in file:
                continue

            # print(file)
            # s = re.sub('[^0-9]','', file)
            chars = 'abcdefghijklmnopqrstuvwxyz.'
            table=str.maketrans("","",chars)
            s = file.translate(table)
            s = s.split(',')
            x = int(s[0])
            y = int(s[1])
            tup = [x,y]
            # segs.append(im[y:y+dim, x:x+dim])
            segs.append(tup)

    return segs
            # print(s)

def display_labels(im, labels):
    lookupTable, idx,  labels1, counts = np.unique(labels, return_inverse=True, return_counts=True, return_index=True)
    print(len(lookupTable), "segments found")

    nrows = math.sqrt(len(lookupTable))
    if nrows % 1 != 0:
        nrows = nrows - nrows % 1
    ncols = len(lookupTable)/nrows


    fig, ax = plt.subplots( int(nrows), int(ncols))
    ax = ax.flatten()
    for i, (label, ax) in enumerate(zip(lookupTable, ax)):
        mask = np.zeros(im.shape[:2], dtype = "uint8")
        mask[labels == label] = 255
        ax.imshow(cv2.bitwise_and(im, im, mask = mask))

    print("--- %s seconds ---" % (time.time() - start_time))
    plt.show()


def main():

    '''
    mode: 1 while looking at full image to find locations to add to coords.
          2 to display selected x,y locations specified in coords to make sure they're correct.
          3 to save segments to training/<your_class>/ after verifiying they look okay.
    path: path to the full image
    dest: path to training folder for respective class
    coords: list of (x, y) location of top left pixel of each segment
    dim: size of chunk side (I think 2 is what we want?)
    '''
    path = "../data/Mac_1120_UTM.tif"
    dest = "p3chunks/bush"
    mode = int(sys.argv[1])
    dim = 30


    road = [
        (3581, 5553),
        (1390, 3886),
        (1980, 4163),
        (1631, 3947),
        (2649, 4641),
        (3192, 5166),
        (3424, 5388),
        (2979, 4929),
        (3890, 5819),
        (4295, 6327),
        (4629, 6739),
        (4856, 7038),
        (5055, 7233),
        (5153, 7309),
        (5373, 7582),
        (5540, 7860),
        (5657, 8098),
        (6890, 9415),
        (7007, 9575),
        (2122, 4244),
        (2556, 4564),
        (2704, 4709),
        (3120, 5065),
        (3715, 5680),
        (4090, 6081),
        (4456, 6476),
        (4985, 7160),
        (4241, 6202),
        (5435, 7706),
        (1258, 3726),
        (1247, 3895),
        (1384, 3983),
        (1506, 3967),
        (2871, 4802),
        (3049, 5003),
        (3237, 5231),
        (3349, 5356),
        (4677, 6806),
        (5214, 7392),
        (5280, 7468),
        (5748, 8226),
        (5596, 7956),
        (1726, 3988),
        (1570, 4031),
        (1724, 3876),
        (1776, 4051),
        ( 1052 , 3737 ),
        ( 1178 , 3825 ),
        ( 1300 , 3933 ),
        ( 1435 , 3950 ),
        ( 1479 , 3892 ),
        ( 1651 , 4015 ),
        ( 1850 , 4094 ),
        ( 1923 , 4137 ),
        ( 2413 , 4467 ),
        ( 2942 , 4873 ),
        ( 3389 , 5314 ),
        ( 2503 , 4566 ),
        ( 3629 , 5617 ),
        ( 3663 , 5581 ),
        ( 4373 , 6414 ),
        ( 3487 , 5506 ),
        ( 3561 , 5477 ),
        ( 3922 , 5876 ),
        ( 4163 , 6179 ),
        ( 4572 , 6720 ),
        ( 4642 , 6693 ),
        ( 4719 , 6893 ),
        ( 4763 , 6879 ),
        ( 5051 , 7181 ),
        ( 5124 , 7255 ),
        ( 5210 , 7337 ),
        ( 5304 , 7543 ),
        ( 5351 , 7514 ),
        ( 5383 , 7652 ),
        ( 5427 , 7637 ),
        ( 5468 , 7781 ),
        ( 5630 , 8025 ),
        ( 5698 , 8172 ),
        ( 5780 , 8313 ),
        ( 3284 , 5289 ),
        ( 3471 , 5469 ),
        ( 3525 , 5509 ),
        ( 3763 , 5727 ),
        ( 3816 , 5776 ),
        ( 3995 , 5954 ),
        ( 6520 , 8954 ),
        ( 6622 , 9047 ),
        ( 6778 , 9247 ),
        ( 6947 , 9467 ),
        ( 7033 , 9640 ),
        ( 7098 , 9764 ),
        (2484, 4501)
    ]
    grass = [
     (2807, 1846),
     (2517, 2237),
     (2739, 2896),
     (2762, 4148),
     (3724,3856),
     (7165,5228),
     (8084,7412),
     (5179,5933),
     (10203,6336),
     (9860, 6951),
     (10052, 7021),
     (8788, 7340),
     (9606, 7686),
     (7301, 8923),
     (9444,8742),
     (7009,9440),
     (7846,10980),
     (8635,10628),
     (9508,11457),
     (8675, 10474),
     (8618, 7156),
     (9618, 6926),
     (8931, 7848),
     (9787, 8750),
     (7624, 10457),
     (11814, 8312),
     (9133, 11777),
     (10416, 9874),
     (9164, 8352),
     (8121, 8863),
     ( 8183 , 7355 ),
    ( 8869 , 7777 ),
    ( 11959 , 7921 ),
    ( 10157 , 8161 ),
    ( 11521 , 8751 ),
    ( 10952 , 8845 ),
    ( 8829 , 9838 ),
    ( 8944 , 9872 ),
    ( 9130 , 10016 ),
    ( 9357 , 9897 ),
    ( 9062 , 10215 ),
    ( 8619 , 10399 ),
    ( 8595 , 10511 ),
    ( 8557 , 10651 ),
    ( 8487 , 10630 ),
    ( 8862 , 10701 ),
    ( 8936 , 10659 ),
    ( 8989 , 10464 ),
    ( 8405 , 10812 ),
    ( 9062 , 10213 ),
    ( 7665 , 10677 ),
    ( 7713 , 10762 ),
    ( 7755 , 10864 ),
    ( 7895 , 11174 ),
    ( 7802 , 11201 ),
    ( 7845 , 11264 ),
    ( 8186 , 10774 ),
    ( 7857 , 10890 ),
    ( 10152 , 11093 ),
    ( 7607 , 10388 ),
    ( 7806 , 10842 ),
    ( 7898 , 11126 ),
    ( 9793 , 9718 ),
    ( 9923 , 9763 ),
    ( 10047 , 9981 ),
    ( 10542 , 9884 ),
    ( 10721 , 9943 ),
    ( 10178 , 10355 ),
    ( 10260 , 10330 ),
    ( 10334 , 10368 ),
    ( 10238 , 10445 ),
    ( 10232 , 10926 ),
    ( 10161 , 10962 ),
    ( 10221 , 11007 ),
    ( 10299 , 10997 ),
    ( 10340 , 11085 ),
    ( 9099 , 11364 ),
    ( 9225 , 11390 ),
    ( 9299 , 11406 ),
    ( 8978 , 11605 ),
    ( 9131 , 11598 ),
    ( 9189 , 11647 ),
    ( 9117 , 11724 ),
    ( 9221 , 11690 ),
    ( 9313 , 11649 ),
    ( 9426 , 11641 ),
    ( 9552 , 11536 ),
    ( 7517 , 8683 ),
    ( 7440 , 9325 ),
    ( 7653 , 9078 ),
     (3647, 2102)
    ]
    saps = [
        ( 5881 , 5280 ),
        ( 6070 , 5172 ),
        ( 5619 , 5258 ),
        ( 5285 , 5166 ),
        ( 5139 , 5070 ),
        ( 5199 , 5015 ),
        ( 5147 , 5299 ),
        ( 5822 , 3877 ),
        ( 5606 , 4140 ),
        ( 4186 , 4408 ),
        ( 4027 , 4803 ),
        ( 3972 , 4883 ),
        ( 4803 , 4295 ),
        ( 4941 , 4269 ),
        ( 5234 , 4370 ),
        ( 4777 , 4504 ),
        ( 4952 , 4490 ),
        ( 5214 , 4477 ),
        ( 4757 , 4700 ),
        ( 5031 , 4786 ),
        ( 4990 , 4713 ),
        ( 5605 , 4139 ),
        ( 5794 , 4182 ),
        ( 5428 , 4192 ),
        ( 5515 , 4221 ),
        ( 5715 , 4525 ),
        ( 6028 , 5264 ),
        ( 6069 , 5170 ),
        ( 5996 , 5117 ),
        ( 5947 , 5220 ),
        ( 5902 , 5126 ),
        ( 5883 , 5279 ),
        ( 5810 , 5156 ),
        ( 5801 , 5263 ),
        ( 5710 , 5204 ),
        ( 5617 , 5258 ),
        ( 5581 , 5360 ),
        ( 5356 , 5057 ),
        ( 5470 , 5157 ),
        ( 5284 , 5165 ),
        ( 5276 , 5090 ),
        ( 5257 , 5355 ),
        ( 5208 , 5151 ),
        ( 5200 , 5015 ),
        ( 5114 , 4864 ),
        ( 5140 , 5071 ),
        ( 5148 , 5174 ),
        ( 5151 , 5297 ),
        ( 5050 , 5302 ),
        ( 5062 , 5225 ),
        ( 4810 , 5275 ),
        ( 9987 , 10976 ),
        ( 10526 , 11423 ),
        ( 10586 , 11365 ),
        ( 10657 , 11336 ),
        ( 10683 , 11558 ),
        ( 10839 , 11472 ),
        ( 10840 , 11576 ),
        ( 11344 , 10754 ),
        ( 4976 , 5348 )
    ]

    bush = [
        ( 12391 , 9527 ),
        ( 12409 , 9553 ),
        ( 12831 , 9615 ),
        ( 12835 , 9645 ),
        ( 13090 , 9015 ),
        ( 13111 , 9031 ),
        ( 12910 , 8309 ),
        ( 12934 , 8327 ),
        ( 11065 , 8446 ),
        ( 11097 , 8460 ),
        ( 9283 , 8035 ),
        ( 9330 , 8054 ),
        ( 9306 , 8075 ),
        ( 4892 , 7477 ),
        ( 4917 , 7478 ),
        ( 4871 , 7514 ),
        ( 4915 , 7515 ),
        ( 4960 , 7515 ),
        ( 4964 , 7551 ),
        ( 4918 , 7555 ),
        ( 4878 , 7550 ),
        ( 4909 , 7578 ),
        ( 4946 , 7581 ),
        ( 6616 , 6813 ),
        ( 6629 , 6832 ),
        ( 6019 , 6753 ),
        ( 6021 , 6767 ),
        ( 10807 , 7200 ),
        ( 3653 , 6197 ),
        ( 3691 , 6200 ),
        ( 3664 , 6227 ),
        ( 3705 , 6231 ),
        ( 3683 , 6261 ),
        ( 3496 , 5741 ),
        ( 3527 , 5739 ),
        ( 3525 , 5721 ),
        ( 3490 , 5685 ),
        ( 3794 , 4942 ),
        ( 3830 , 4942 ),
        ( 3846 , 4964 ),
        ( 5503 , 5873 ),
        ( 5541 , 5877 ),
        ( 5499 , 5904 ),
        ( 5542 , 5902 ),
        ( 5522 , 5924 ),
        ( 8988 , 5275 ),
        ( 9013 , 5284 ),
        ( 2880 , 4462 ),
        ( 2914 , 4477 ),
        ( 2909 , 4505 ),
        ( 2967 , 4672 ),
        ( 3018 , 4683 ),
        ( 3050 , 4670 ),
        ( 3571 , 4776 ),
        ( 3571 , 4801 ),
        ( 3801 , 4903 ),
        ( 3852 , 4906 ),
        ( 3805 , 4940 ),
        ( 3848 , 4954 ),
        ( 4088 , 4217 ),
        ( 4099 , 4278 ),
        ( 4040 , 4256 ),
        ( 4127 , 4252 ),
        ( 4667 , 4853 ),
        ( 4716 , 4838 ),
        ( 4674 , 4801 ),
        ( 4726 , 4783 ),
        ( 4700 , 4821 ),
        ( 4877 , 4863 ),
        ( 4934 , 4877 ),
        ( 4895 , 4912 ),
        ( 5266 , 4582 ),
        ( 5298 , 4553 ),
        ( 5894 , 4332 ),
        ( 5883 , 4322 ),
        ( 5907 , 4349 ),
        ( 6159 , 4539 ),
        ( 6182 , 4549 ),
        ( 6171 , 4515 ),
        ( 6179 , 4494 ),
        ( 7700 , 3953 ),
        ( 7724 , 3939 ),
        ( 7727 , 3925 ),
        ( 8997 , 5278 ),
        ( 3496 , 5680 ),
        ( 3523 , 5728 ),
        ( 3513 , 5744 ),
        ( 3662 , 6206 ),
        ( 3683 , 6231 ),
        ( 3697 , 6253 ),
        ( 1086 , 5186 ),
        ( 1108 , 5225 ),
        ( 1433 , 4885 ),
        ( 1450 , 4857 ),
        ( 1408 , 4857 ),
        ( 1420 , 4833 )

    ]

    coords = grass

    rgb = imread(path)
    rgb = (rgb/65535).astype('float32') #for plantation 3 color depth


    print("saps: ", len(saps))
    print("road: ", len(road))
    print("grass: ", len(grass))
    print("bush: ", len(bush))



    if mode == 1:
        coords = read_coords("../plant3/")
        all_coords = coords + road + grass + saps
        im = mark_img(rgb, all_coords)


        f, axes = plt.subplots(1,1)
        axes.imshow(im)

        cid = f.canvas.mpl_connect('button_press_event', onclick)
        plt.show()


    elif mode == 2:
        segs = []
        print(len(coords))
        coords = bush

        for coord in coords:
            cropped = crop(rgb, coord[0], coord[1], dim)
            segs.append(cropped)
        verify_chunks(segs)

    elif mode == 3:
            print(len(coords))
            coords = bush

            for coord in coords:
                crop_and_write_from_mid(rgb, coord[0], coord[1], dim, dest)




main()
