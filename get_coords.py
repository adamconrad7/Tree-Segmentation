from tifffile import imread, imwrite
import cv2
from PIL import Image

def crop(img, x, y, len_x, len_y):
    cropped = img[y:y+len_y, x:x+len_x]
    cv2.imshow("cropped", cropped)
    #cv2.waitKey(0)
    return cropped

def main():
    ## Path to data
    path = "data/plantation1.tif"
    ## Read data
    rgb0 = imread(path)
    cropped = crop(rgb0, 3000, 500, 1500, 1500)
    

main()