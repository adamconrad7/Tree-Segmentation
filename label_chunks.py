import matplotlib.pyplot as plt
from tifffile import imread, imwrite

def crop(img, x, y, dim, dest):
    new = img[y:y+dim, x:x+dim]
    imwrite(dest, new)
    print(new.shape)
    return new

def main():
    '''
    path: path to the full image
    dest: path to training folder for respective class
    x, y: location of top left pixel in segment
    dim: size of chunk side (I think 2 is what we want? Up to Jaelyn)
    '''

    path = "data/plantation1.tif"
    dest = "training/strimbu/bogdan.tif"
    x = 3200
    y = 2700
    dim = 200

    rgb = imread(path)
    croped = crop(rgb, x, y, dim, dest)

    f, ax = plt.subplots(1,2)
    ax[0].imshow(rgb)
    ax[1].imshow(croped)
    plt.show()

main()
    #
