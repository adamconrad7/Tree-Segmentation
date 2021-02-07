import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from os import listdir
from os.path import isfile, join
import skimage.segmentation as seg
import skimage.color as color
import skimage.future.graph as graph
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches

import cv2

import numpy as np
import os
import time
import math

start_time = time.time()



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

    # path = "data/plantation1.tif"
    dest = "plantation3/grass/"
    path = "data/Mac_1120_UTM.tif"
    mode = 2
    dim = 30
    # coords = [
    #     (3210, 9388),
    #     (3266, 9639),
    #     (3423, 9449),
    #     (3235, 10076),
    #     (3830, 9351),
    #     (3844, 9565),
    #     (4820, 9651),
    #     (5007, 9815),
    #     (5233, 9827),
    #     (5718, 9914),
    #     (4781, 10043),
    #     (4976, 10112),
    #     (5588, 10051),
    #     (5741, 10142),
    #     (5207, 10428),
    #     (5436, 10370),
    #     (7245, 9270),
    #     (8310, 9568),
    #     (7221, 9714),
    #     (7822, 9869),
    #     (7271, 10131),
    #     (3171, 10472),
    #     (3543, 10410),
    #     (3130, 10670),
    #     (3340, 10659),
    #     (3546, 10613),
    #     (3383, 10827),
    #     (3787, 10864),
    #     (4440, 11169),
    #     (5091, 11318),
    #     (5030, 11505),
    #     (5265, 11574),
    #     (4881, 11611),
    #     (6521, 10681),
    #     (5863, 10751),
    #     (6577, 10875),
    #     (5800, 11177),
    #     (6620, 11058),
    #     (6651, 11304),
    #     (6993, 10579),
    #     (7152, 10636),
    #     (7821, 10774),
    #     (7021, 11090),
    #     (7800, 11156),
    #     (7801, 11463),
    #     (6877, 11539),
    #     (3960, 12520),
    #     (4336, 12391),
    #     (4888, 12498),
    #     (5065, 12377),
    #     (5398, 12253),
    #     (4887, 12500),
    #     (6902, 12078),
    #     (6917, 12267),
    #     (7631, 12468),
    #     (5148, 10063),
    #     (6146, 9890),
    #     (6367, 9881),
    #     (6245, 10049),
    #     (6085, 10111),
    #     (6249, 10275),
    #     (6401, 10400),
    #     (5042, 10608),
    #     (2981, 11550),
    #     (3153, 11509),
    #     (3345, 11461),
    #     (3262, 11294),
    #     (3365, 11675),
    #     (2842, 11959),
    #     (3183, 12133),
    #     (3399, 12851),
    #     (7422, 9755),
    #     (3827, 12834),
    #     (4000, 12935),
    #     (6820, 9101),
    #     (7023, 9119),
    #     (7215, 9052),
    #     (4997, 10784),
    #     (3656, 10993),
    #     (4998, 12010),
    #     (7530, 11176),
    #     (7156, 11432),
    #     (3646, 641),
    #     (3665, 765),
    #     (3682, 878),
    #     (3695, 991),
    #     (3691, 1096),
    #     (3683, 1193),
    #     (3740, 1448),
    #     (3867, 630),
    #     (3934, 1189),
    #     (4019, 1002),
    #     (4047, 1195),
    #     (4121, 711),
    #     (4136, 942),
    #     (4149, 1056),
    #     (4148, 1157),
    #     (4190, 1409),
    #     (4184, 586),
    #     (4229, 698),
    #     (4252, 935),
    #     (4341, 698),
    #     (4349, 807),
    #
    #     (7626, 12668),
    #     (3340,4591),
    #     (3354,4860),
    #     (3520,5288),
    #     (3507,5054),
    #     (2993,4911),
    #     (2973,5114),
    #     (3123,5104),
    #     (2786,5140),
    #     (2947,5300),
    #     (3181,4880),
    #     (4030,5456),
    #     (4017,5617),
    #     (4270,5365),
    #     (4224,5569),
    #     (4480,5572),
    #     (4675,5365),
    #     (5562,5151),
    #     (5582,5573),
    #     (3238,6168),
    #     (3606,6303),
    #     (5136,6040),
    #     (3686,6488),
    #     (4021,6108),
    #     (4190,5946),
    #     (4227,6117),
    #     (3732,5981),
    #     (4281,6359),
    #     (4655,6474),
    #     (3771,5783),
    #     (3484,6478),
    #     (2705,6600),
    #     (2903,6588),
    #     (3522,6718),
    #     (4420,6772),
    #     (4667,7047),
    #     (4460,7058),
    #     (3069,7019),
    #     (2730,7191),
    #     (2295,7400),
    #     (2240,7211),
    #     (2564,7438),
    #     (2753,7620),
    #     (2997,7618),
    #     (3105,7431),
    #     (4712,7443),
    #     (5190,7260),
    #     (2405,7610),
    #     (2924,7797),
    #     (2108,7741),
    #     (2303,7741),
    #     (3046,8091),
    #     (3212,8068),
    #     (4325,7962),
    #     (4539,7986),
    #     (4777,7846),
    #     (5191,7738),
    #     (5326,7978),
    #     (5477,7910),
    #     (5250,8238),
    #     (5426,8220),
    #     (4378,8216),
    #     (2856,8670),
    #     (3008,8490),
    #     (3276,8297),
    #     (3369,8278),
    #     (3479,8206),
    #     (3410,8502),
    #     (3301,8786),
    #     (3350,8998),
    #     (3859,9031),
    #     (4573,9098),
    #     (4411,8962),
    #     (4410,8529),
    #     (5249,8239),
    #     (5298,8652),
    #     (5424,8223),
    #     (6167,8333),
    #     (6951,8910),
    #     (7022,9120),
    #     (7220,9051),
    #     (6820,9110),
    #     (7420,8970),
    #     (7780,8883)
    # ]
    # non_coords = [
    #     (1398, 1192),
    #     (1883, 670),
    #     (2766, 932),
    #     (1961, 1383),
    #     (1688, 1736),
    #     (4083, 738),
    #     (2608, 2444),
    #     (1665, 1883),
    #     (1275, 2595),
    #     (3477, 1970),
    #     (2245, 1346),
    #     (1781, 2580),
    #     (4056, 736),
    #     (4545, 628),
    #     (4995, 647),
    #     (6427, 723),
    #     (4807, 1315),
    #     (5421, 1343),
    #     (5850, 1537),
    #     (4807, 1903),
    #     (6413, 1939),
    #     (4973, 2165)
    # ]

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

     (3647, 2102)
    ]
    coords = grass

    rgb = imread(path)
    rgb = (rgb/256).astype('uint8') #for plantation 3 color depth


# print(rgb.shape[0])
    # print(rgb.shape[0]/3)
    # print((rgb.shape[0]/3)*2)
    # print(rgb.shape[0])


    if mode == 1:
        im = mark_img(rgb, coords)
        f, axes = plt.subplots(1,1)
        axes.imshow(im)
        plt.show()


        # segs = read_segs('training')
        # f, axes = plt.subplots(6,25)
        # for i, ax in enumerate(axes.ravel()):
        #
        #     ax.imshow(segs[i,0])
        #     ax.axes.xaxis.set_ticks([])
        #     ax.axes.yaxis.set_ticks([])
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        # plt.show()
        # print(segs.shape)
        # classes = np.unique(segs[:,1])
        # mask = segs[:,1] == classes[0],
        # # print(segs[mask])
        # for label in classes:
        #     mask = segs[:,1] == label
        #     chunks = segs[mask]
            # print(chunks[:,0])
            # verify_chunks(chunks[:,0])
            # print(segs[mask])

        # find_xy(rgb)
    elif mode == 2:
        segs = []
        for coord in coords:
            cropped = crop(rgb, coord[0], coord[1], dim)
            segs.append(cropped)
        verify_chunks(segs)
    elif mode == 3:
        for coord in coords:
            crop_and_write(rgb, coord[0], coord[1], dim, dest)

    elif mode == 4:
        print(len(coords))

        for coord in coords:
            crop_and_write_from_mid(rgb, coord[0], coord[1], dim, dest)

    elif mode == 5:
        segs = chunk_from_coords('training', rgb, 50)
        print(len(non_coords))
        non_coords = non_coords + segs
        print(len(non_coords))
        for coord in non_coords:
            crop_and_write(rgb, coord[0], coord[1], 50, dest)


        # # fig, ax = plt.subplots(1,1)
        # # ax.imshow(segs[0])
        # print(len(segs))
        #
        # fig, axs = plt.subplots(10, int(len(segs) / 10))
        # axs = axs.flatten()
        # for img, ax in zip(segs, axs):
        #     ax.imshow(img)
        # plt.show()


    else:
        # m = mark_img(rgb, coords)
    # rgb = downscale(rgb0[:int(rgb0.shape[0]/3), int(rgb0.shape[1]/3):], downscale_percent)

        rgb = rgb[:int(rgb.shape[0]/3), int(rgb.shape[1]/3):]

        labels1 = seg.slic(rgb[...,:-1],compactness=.01, n_segments=8, sigma=2, convert2lab=True, enforce_connectivity=False, start_label= 1)

        # labels1[labels1 != 1] = 2

        # display_labels(rgb, labels1)


        fig, ax = plt.subplots(figsize=(10, 6))

        for i, region in enumerate(regionprops(labels1)):
            print(i)
            # take regions with large enough areas
            if region.area >= 1:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

        ax.set_axis_off()
        plt.tight_layout()
        plt.show()




main()
