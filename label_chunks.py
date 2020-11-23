import matplotlib.pyplot as plt
from tifffile import imread, imwrite
from os import listdir
from os.path import isfile, join
import numpy as np
import os



def crop(img, x, y, dim):
    return img[y:y+dim, x:x+dim]

def crop_and_write(img, x, y, dim, dest):
    new = img[y:y+dim, x:x+dim]
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

    path = "data/plantation1.tif"
    dest = "training/<your_class>/"
    mode = 1
    dim = 2
    coords = [
        (1665, 1919),
        (1774, 2563),
        (3667, 1719),
        (1978, 3244),
        (3325, 2494),
        (3542, 2778),
        (3429, 11608),
        (3582, 12009),
        (3674, 12536),
        (3927, 12873),
        (3945, 12962),
        (4103, 13282),
        (5127, 361),
        (6480, 741),
        (6430, 1955),
        (2989, 6105),
        (4265, 6031),
        (3421, 5820),
        (3419, 5984),
        (1012, 3867),
        (1975, 3357),
        (1046, 4607),
        (1466, 4047),
        (1251, 5213),
        (1554, 3849)
    ]

    rgb = imread(path)

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
            cropped = crop(rgb, coord[0], coord[1], 2)
            segs.append(cropped)
        verify_chunks(segs)
    else:
        for coord in coords:
            crop_and_write(rgb, coord[0], coord[1], 2, dest)


main()
