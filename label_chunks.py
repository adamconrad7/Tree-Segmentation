import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from os import listdir
from os.path import isfile, join
import numpy as np
import os

import cv2
def downscale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # print("dim: ", width, height)

    r = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return r

def crop(img, x, y, dim):
    return img[y:y+dim, x:x+dim]

def crop_centered(img, x, y, dim): #dim is total height (must be even)
    return img[int(y-(dim/2)):int(y+(dim/2)), int(x-(dim/2)):int(x+(dim/2))]

def crop_and_write(img, x, y, dim, dest):
    new = img[y:y+dim, x:x+dim]
    imwrite(dest + str(x) + ',' + str(y) + '.tif', new)
    return new

def crop_and_write_centered(img, x, y, dim, dest):
    new = img[int(y-(dim/2)):int(y+(dim/2)), int(x-(dim/2)):int(x+(dim/2))]
    imwrite(dest + str(x) + ',' + str(y) + '.tif', new)
    return new

def find_xy(im):
    f, axes = plt.subplots(1,1)
    axes.imshow(im)
    plt.show()

def verify_chunks(segs):
    f, axes = plt.subplots(5,5)
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(segs[i])
    plt.show()

def read_segs(path):
    # onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # for file in listdir('training/bark/'):
    #     print(file)
    # print(listdir(path))
    segs = []
    for classname in listdir(path):
        classpath = os.path.join(path, classname)
        for file in listdir(classpath):
            seg = imread(os.path.join(classpath, file))
            tup = [seg, classname]
            segs.append(tup)

    return np.array(segs)
            # print(file)


        # if not listdir(classpath):
        #     print("{} is empty", classpath)
        # else:
        #     for file in listdir(classpath):
        #         print(file)

    # print(onlyfiles)


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

    path = "D:/College Documents/Senior Design/Mac_1120_UTM.tif"
    dest = "training/plantation3/<your_class>/"
    mode = 3
    dim = 30
    coords = [(9406, 8965),
(9619, 8894),
(9821, 9268),
(9320, 9204),
(9538, 9389),
(10001, 10239),
(9803, 10259),
(10106, 9671),
(9853, 9609),
(10070, 9808),
(3083, 3975),
(2943, 4010),
(3210, 4086),
(3154, 3807),
(3380, 4143),
(2442, 4174),
(2349, 4041),
(2224, 3949),
(2709, 4319),
(2636, 4379),
(2623, 3773),
(7994, 5511),
(8045, 5440),
(8466, 5218),
(8593, 5481),
(8759, 5408),
(9053, 5501),
(8968, 5182),
(8132, 6211),
(8333, 6213)]

    rgb = imread(path)
    #rgb = downscale(rgb, 20) #for plantation 3's size
    rgb = (rgb/256).astype('uint8') #for plantation 3 color depth

    if mode == 1:
        f, axes = plt.subplots(1,1)
        axes.imshow(rgb)
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
            #cropped = crop(rgb, coord[0], coord[1], 2)
            cropped = crop_centered(rgb, coord[0], coord[1], 30)
            segs.append(cropped)
        verify_chunks(segs)
    else:
        for coord in coords:
            crop_and_write_centered(rgb, coord[0], coord[1], 30, dest)


main()
