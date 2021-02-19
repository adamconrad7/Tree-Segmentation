from tifffile import imread, imwrite
import cv2
from PIL import Image
import numpy as np
from tensorflow import keras
import sys
from matplotlib import colors
import matplotlib.pyplot as plt

from scipy import ndimage
#from treeSeg import chunkify

#returns and displays cropped image
def crop(img, x, y, len_x, len_y):
    cropped = img[y:y+len_y, x:x+len_x] #crops
    cv2.imshow("cropped", cropped) #shows the cropped section
    cv2.waitKey(0) #press 0 key to continue processing
    return cropped

#returns 2x2x4 np array of pixels and their rgba values
def get_rgbs(img):
    cropped = np.array(img)
    cropped = cropped/ 255. #accuracy thing done in classifier.py

    test = []

    for i in range(0, len(cropped), 2): #skips one for 2x2
        for j in range(0, len(cropped[i]), 2): #skips one for 2x2
            
            p1 = np.array([cropped[i][j], cropped[i][j+1]])
            p2 = np.array([cropped[i+1][j], cropped[i+1][j+1]])
            xs = np.array([p1, p2])
            #xs = [[cropped[i][j], croppsed[i][j+1]]]
            #xs.append([[cropped[i+1][j]], [cropped[i+1][j+1]]])
            #xs = np.array(xs)
            test.append(xs)
        
    test = np.array(test) #formats data correctly for the model
    return test

#creates and displays the classification plot
def plot_color(matrix):
    #data = np.random.rand(10, 10) * 20

    # create discrete colormap
    cmap = colors.ListedColormap(['black', 'yellow', 'lime', 'red', 'blue', 'white'])
    bounds = [0, 1, 2, 3, 4, 5] #for the 0-5 classes (?)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    blurred = ndimage.median_filter(matrix, size=7) #filters noise out of classification - removes lone class boxes

    fig, ax = plt.subplots()
    ax.imshow(blurred)

    # draw gridlines
    ax.grid(which='major', axis='both', color='k')
    ax.set_xticks(np.arange(0, 100, 10)) #change with size of cropped image
    ax.set_yticks(np.arange(0, 100, 10)) #change with size of cropped image

    plt.show()

#turn the list of predictions into a matrix (half) the dimensions of the cropped image (half bc 2x2 chunks)
def manual_as_matrix(classes):
    mat = np.mat(classes)
    mat = mat.reshape(100, 100) #change with size of cropped image
    print(mat)
    return mat


def main():
    ## Path to data
    path = "data/plantation1.tif"
    ## Read data
    rgb0 = imread(path)

    cropped = crop(rgb0, 3500, 500, 200, 200) #x coord, y coord, x length, y length
    #og: 3500, 500, 200, 200 //1500, 1500

    #cropped = np.array(cropped)
    #print(len(cropped))
    #print(len(cropped[0]))
    print('len of cropped: ' + str(len(cropped)))
    print('len of cropped[0]: ' + str(len(cropped[0])))
    test = get_rgbs(cropped) #nparray (2x2x4) of rgb in 4 pixel chunks 
    print('len of test: ' + str(len(test)))
    print('len of test[0]: ' + str(len(test[0])))
    print('test[0]: ', end='')
    print(test[0])

    model = keras.models.load_model('model/') #load model saved from classifier.py
    #model = keras.models.load_model('model/model_1.h5') #load model saved from classifier.py


    y_pred = model.predict_classes(test) #run classifier on the 4 pixel chunks, returns numerical class prediction for each chunk
    print(y_pred.shape)
    np.set_printoptions(threshold=sys.maxsize) #prints the whole np array
    print(y_pred)
    print(len(y_pred))

    mat = manual_as_matrix(y_pred) #turn list of predicted classes into the dimensions of the cropped image
    plot_color(mat) #plots color representation of predictions on a graph

main()
