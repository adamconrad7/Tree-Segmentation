from tifffile import imread, imwrite
import cv2
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf
import sys
from matplotlib import colors
import matplotlib.pyplot as plt

from scipy import ndimage
#from treeSeg import chunkify

#returns and displays cropped image
def crop(img, x, y, len_x, len_y):
    cropped = img[y:y+len_y, x:x+len_x] #crops
    #cv2.imshow("cropped", cropped) #shows the cropped section
    #cv2.waitKey(0) #press 0 key to continue processing
    return cropped

#returns 30x30x4 np array of pixels and their rgba values
def get_rgbs(img, dim):
    cropped = np.array(img)
    cropped = cropped/ 255. #accuracy thing done in classifier.py

    test = [] #list of the 30x30 chunks

    for j in range(0, len(cropped), dim): #skips 30 for 30x30 (iterates cols)
        #print('i: ' + str(i))
        
        
        for i in range(0, len(cropped[j]), dim): #iterates rows
            rows = []
            for k in range(0, dim):
                small_row = cropped[k+j][i:i+dim]
                rows.append(small_row)
            
            test.append(rows)
        
    test = np.array(test) #formats data correctly for the model
    return test

#creates and displays the classification plot
def plot_color(matrix, by_x, by_y):
    #data = np.random.rand(10, 10) * 20

    # create discrete colormap
    cmap = colors.ListedColormap(['black', 'yellow', 'lime', 'red', 'blue', 'white'])
    bounds = [0, 1, 2, 3, 4, 5] #for the 0-5 classes (?)
    norm = colors.BoundaryNorm(bounds, cmap.N)

    #blurred = ndimage.median_filter(matrix, size=7) #filters noise out of classification - removes lone class boxes

    fig, ax = plt.subplots()
    #ax.imshow(blurred)
    ax.imshow(matrix)

    # draw gridlines
    ax.grid(which='major', axis='both', color='k')
    ax.set_xticks(np.arange(0, by_x, 5)) #change with size of cropped image
    ax.set_yticks(np.arange(0, by_y, 5)) #change with size of cropped image

    plt.show()

#turn the list of predictions into a matrix (half) the dimensions of the cropped image (half bc 2x2 chunks)
def manual_as_matrix(classes, x, y):
    mat = np.mat(classes)
    mat = mat.reshape(x, y) #change with size of cropped image
    print(mat)
    return mat

def get_sapling_indices(matrix, start_x, start_y):
    inds = np.where(matrix == 2) #[0] is the list of row indexes, [1] is the list of column indexes
    print(inds)
    xs = inds[0]
    ys = inds[1]
    #print(inds[0])
    #print(type(inds[0]))
    #convert coordinates from row column, to pixels in image
    xs = (xs * 30) + start_x + 15 #*30 to get to correct box, +start to get to correct region, +15 to set coordinate at middle of 30x30 box
    ys = (ys * 30) + start_y + 15
    print(inds)
    img_coords = list(zip(xs, ys))
    int_coords = list(zip(inds[0], inds[1]))
    print(img_coords)
    print("number of saplings: " + str(len(img_coords)))
    for c in img_coords:
        #print(type(c))
        print('(' + str(c[0]) + ', ' + str(c[1]) + ')')
        #print(matrix.item(c))
        #print(matrix[c[0]][c[1]])
    return img_coords, int_coords

def mark_saplings(cropped, mat_coords, x, y):
    for c in mat_coords:
        print('(' + str(c[0]-x) + ', ' + str(c[1]-y) + ')')
        image = cv2.circle(cropped, (c[1]-y,c[0]-x), radius=5, color=(0, 0, 255), thickness=-1)
    
    cv2.imshow("sapling_labeled", image) #shows the cropped section
    cv2.waitKey(0)

def main():
    ## Path to data
    path = "D:/College Documents/Senior Design/Mac_1120_UTM.tif" #"data/plantation1.tif"
    ## Read data
    rgb0 = imread(path)

    coo = []

    width = 900 #must be divisible by 30 for now
    height = 900 #must be divisible by 30 for now
    dim = 30
    x_start = 2220
    y_start = 3900

    for itr in range(0, 12):
        if itr < 6:
            x = x_start #+ itr
            y = y_start + ((itr % 6) * 5)
        else:
            x = x_start + ((itr % 6) * 5)
            y = y_start #+ itr
        '''if itr == 1:
            x += 15
        elif itr == 2:
            x -= 15
        elif itr == 3:
            y += 15
        elif itr == 4:
            y -= 15
        elif itr == 5:
            x += 15
            y += 15
        elif itr == 6:
            x += 15
            y -= 15
        elif itr == 7:
            x -= 15
            y += 15
        elif itr == 8:
            x -= 15
            y -= 15
        '''

        
        cropped = crop(rgb0, x, y, width, height) #x coord, y coord, x length, y length
        cropped = (cropped/256).astype('uint8') #for plantation 3 color depth

        #cropped = crop(rgb0, 3500, 500, 200, 200) #x coord, y coord, x length, y length
        #og: 3500, 500, 200, 200 //1500, 1500

        #cropped = np.array(cropped)
        #print(len(cropped))
        #print(len(cropped[0]))
        #print(cropped)
        print('len of cropped: ' + str(len(cropped)))
        print('len of cropped[0]: ' + str(len(cropped[0])))
        print('len of cropped[0][0]: ' + str(len(cropped[0][0])))
        test = get_rgbs(cropped, dim) #nparray (2x2x4) of rgb in 4 pixel chunks 
        print(test[0])
        #print(test[0][0])
        print('len of test: ' + str(len(test)))
        print('len of test[0]: ' + str(len(test[0])))
        #print('len of test[0][0]: ' + str(len(test[0][0])))

        model = keras.models.load_model('model/30x30model.h5') #load model saved from classifier.py
        #model = keras.models.load_model('model/model_1.h5') #load model saved from classifier.py
        

        y_pred = model.predict_classes(test) #run classifier on the 4 pixel chunks, returns numerical class prediction for each chunk
        print(y_pred.shape)
        np.set_printoptions(threshold=sys.maxsize) #prints the whole np array
        print(y_pred)
        print(len(y_pred))

        by_x = int(width/dim)
        by_y = int(height/dim)
        print('by_x: ' + str(by_x))
        print('by_y: ' + str(by_y))
        mat = manual_as_matrix(y_pred, by_x, by_y) #turn list of predicted classes into the dimensions of the cropped image
        #plot_color(mat, by_x, by_y) #plots color representation of predictions on a graph

        coords, i_coords = get_sapling_indices(mat, x, y)
        for c in coords:
            coo.append(c)
        #mark_saplings(cropped, coords, i_coords, x, y)
    coo = sorted(coo)
    print(coo)

    cropped = crop(rgb0, x_start, y_start, width+30, height+30) #x coord, y coord, x length, y length
    cropped = (cropped/256).astype('uint8')
    mark_saplings(cropped, coo, x_start, y_start)
main()
