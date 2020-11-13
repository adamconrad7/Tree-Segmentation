import matplotlib.pyplot as plt
from tifffile import imread, imwrite

def crop(img, x, y, dim):
    return img[y:y+dim, x:x+dim]

def crop_and_write(img, x, y, dim, dest):
    new = img[y:y+dim, x:x+dim]
    imwrite(dest + str(x)[0:2] + str(y)[0:2] + '.tif', new)
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
    dest = "training/road/"
    mode = 1
    dim = 2
    coords = [
        (2100, 3400),
        (2300, 4200),
        (2200, 4200),
        (2400, 6000),
        (1600, 8000),
        (2200, 5800),
        (1950, 1400),
        (1900, 800),
        (2000, 1400),
        (2120, 1300),
        (2130, 1900),
        (2200, 6000),
        (2100, 3000),
        (2100, 4000),
        (2200, 4000),
        (2200, 5000),
        (2300, 5000),
        (2100, 6000),
        (1900, 7000),
        (2100, 3100),
        (2100, 3300),
        (2200, 3300),
        (2300, 3400),
        (2150, 3000),
        (2260, 3008)
    ]

    rgb = imread(path)

    if mode == 1:
        find_xy(rgb)
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
