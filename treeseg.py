import math
from math import sqrt
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import skimage.segmentation as seg
from skimage import measure
from scipy import ndimage
from itertools import count, islice


class Data:

    def __init__(self, imPath, coordsPath, dim):
#         rgb = imread(imPath)
#         self.im = (rgb/65535).astype('float32')
        self.im = imPath
        self.coords = self.read_coords(coordsPath)
        self.dim = dim
        self.samples = self.read_and_crop()

    def crop(self, x, y):
        xstart = int(x - (self.dim/2))
        ystart = int(y - (self.dim/2))
        xend = int(x + (self.dim/2))
        yend = int(y + (self.dim/2))
        return self.im[ystart:yend, xstart:xend]

    def crop_corner(self, x, y, dim):
        return self.im[y:y+dim, x:x+dim]

    def _crop(self, im, x, y):
        xstart = int(x - (self.dim/2))
        ystart = int(y - (self.dim/2))
        xend = int(x + (self.dim/2))
        yend = int(y + (self.dim/2))
        return im[ystart:yend, xstart:xend]

    def crop_ims(self, big, coords):
        candidates =[]
        for c in coords:
            xstart = np.round((c[1]+self.dim/2) - (self.dim/2)).astype('int')
            xend = np.round((c[1]+self.dim/2) + (self.dim/2)).astype('int')
            ystart = np.round((c[0]+self.dim/2) - (self.dim/2)).astype('int')
            yend = np.round((c[0]+self.dim/2) + (self.dim/2)).astype('int')
            candidates.append(big[ystart:yend, xstart:xend])
        return np.array(candidates)

    def read_coords(self, path):
        with open(path, 'r') as f:
            existing = json.load(f)
        return existing

    def read_and_crop(self):
        ims = []
        labs = []
        for key in self.coords.keys():
            for v in self.coords[key]:
                cropped = self.crop(v[0], v[1])
                ims.append(cropped)
                labs.append(key)
        return [np.array(ims), np.array(labs)]

    def plot(self, ims):
        if isinstance(ims, np.ndarray):
            plt.imshow(ims)
            return
    #     ims = list(ims)
        n = len(ims)

        x = 0
        y = 0
        factors =0
        if n > 1 and all(n % i for i in islice(count(2), int(sqrt(n)-1))):
            if n > 7:
                print("prime!!")
                n += 1
                factors = list(sorted(set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))
                if len(factors) % 2 != 0:
                    x = factors[(int(len(factors) / 2))]
                    y = factors[(int(len(factors) / 2))]

                else:
                    x= factors[int(len(factors) / 2)-1]
                    y=factors[int(len(factors) / 2)]
            else:
                x = 1
                y = n
        else:
            factors = list(sorted(set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))
            if len(factors) % 2 != 0:
                x = factors[(int(len(factors) / 2))]
                y = factors[(int(len(factors) / 2))]

            else:
                x = factors[int(len(factors) / 2)-1]
                y = factors[int(len(factors) / 2)]

    #     print(factors)
    #     print(x, y)
        fig, axs = plt.subplots(x, y)
        axs = axs.flatten()
        for i, (ax, im) in enumerate(zip(axs, ims)):
            ax.imshow(im)
    #         ax.imshow(im, cmap='gray')

            ax.axis('off')
        plt.show()


# In[4]:


class Train(Data):
    def verify_chunks(self):
        segs = self.read_and_crop()[0]
        nrows = math.sqrt(len(segs))
        if nrows % 1 != 0:
            nrows = nrows - nrows % 1
        ncols = len(segs)/nrows
        fig, axs = plt.subplots( int(nrows), int(ncols))
        axs = axs.flatten()
        for i, (ax, seg) in enumerate(zip(axs, segs)):
            ax.imshow(seg)
            # ax.set_title([])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
#             ax.set_title(i)
        plt.show()

    def onclick(self, event):
        if event.dblclick:
            print("[\n",int(event.xdata),",\n",int(event.ydata),"\n],")

    def mark_img(self):
        dim = 10
        nim = self.im.copy()
        for key in self.coords.keys():
            for coord in self.coords[key]:
                xstart = int(coord[0] - (dim/2))
                ystart = int((coord[1]) - (dim/2))
                xend = int(coord[0] + (dim/2))
                yend = int((coord[1]) + (dim/2))
                mark = np.full((4), [255, 0, 0, 255])
                nim[ystart:yend, xstart:xend] = mark
        return nim

    def label(self):
        im = self.mark_img()
        f, axes = plt.subplots(1,1)
        axes.imshow(im)
        cid = f.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()




# In[5]:


class Model(Data):
    def augment(self, labels):
        x = []
        y = []
        for i in range(0, len(self.samples[0])):
            r90 = np.rot90(self.samples[0][i])
            r180 = np.rot90(r90)
            r270 = np.rot90(r180)
            x.append(self.samples[0][i])
            x.append(r90)
            x.append(r180)
            x.append(r270)
            x.append(np.flipud(self.samples[0][i]))
            x.append(np.fliplr(self.samples[0][i]))
            x.append(np.flipud(r90))
            x.append(np.flipud(r180))
            for j in range(0, 8):
                y.append(labels[i])
        return [np.asarray(x), np.asarray(y)]

    def check_data(self):
        nrows = math.sqrt(len(self.samples[0]))
        if nrows % 1 != 0:
            nrows = nrows - nrows % 1
        ncols = len(self.samples[0])/nrows

        fig, axs = plt.subplots( int(nrows), int(ncols))
        axs = axs.flatten()
        for i, (ax, seg) in enumerate(zip(axs, self.samples[0])):
            ax.imshow(seg)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        plt.show()

    def plot_metric(self, history, metric):
        train_metrics = history.history[metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics)
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric])
        plt.show()

    def train(self):
        lookupTable, idx,  labels, counts = np.unique(self.samples[1], return_inverse=True, return_counts=True, return_index=True)


        ## 80% of data for training
        train_test_split = .9

        ## 10% of training data for validation
        validation_split = .3

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            min_delta=0.0005,
            mode='auto'
        )

        ## Adds data
        x, y = self.augment(labels)
        # print(x.dtype)

        ## Shuffles data
        # idx = np.random.permutation(len(x))
        idx = np.random.RandomState(seed=7).permutation(len(x))
        X,Y = x[idx], y[idx]
        # # X,Y = x, y

        print(X.dtype)
        print(X.shape)

        print(Y.dtype)
        print(Y.shape)
        print(Y[0])




        # Define per-fold score containers
        acc_per_fold = []
        loss_per_fold = []


        x_train, x_test= X[:int(X.shape[0]*train_test_split)], X[int(X.shape[0]*train_test_split):]
        y_train, y_test = Y[:int(Y.shape[0]*train_test_split)], Y[int(Y.shape[0]*train_test_split):]


        model = Sequential()
        model.add(Conv2D(8, kernel_size=3, activation='relu', input_shape=(32,32,4)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same',strides=2)),
        model.add(Conv2D(16, kernel_size=5, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same',strides=2)),
        model.add(Conv2D(32, kernel_size=5, activation='relu')),
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same',strides=2)),
        model.add(Flatten()),
        model.add(Dense(len(lookupTable), activation='softmax'))



        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), #change for tf v1 compat
            optimizer = 'adam',    #.97-.98
            metrics = ['accuracy']
        )


        history = model.fit(
            x_train, y_train,
            epochs = 10,
        #         validation_split=validation_split,
            callbacks=[early_stopping]
        )


        self.plot_metric(history, 'accuracy')



        print("Evaluating: \n\n\n")
        metrics = model.evaluate(x_test, y_test)

#         y_pred = model.predict_classes(x_test)
        y_pred = np.argmax(model.predict(x_test), axis=-1)
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred.round(), normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lookupTable)
        print(lookupTable)
        disp = disp.plot(cmap='gray')
        plt.show()

        self.model = model
        # if metrics[1] > .985:
        #     model.save('model5classes/')

        print(len(y_test))


# In[6]:


class Segmenter(Data):
    def __init__(self, rgb, coords, dim, model):
        self.model = model
#         imPath, coordsPath, dim
        super().__init__(rgb, coords, dim)

    def centers(self, im):
        input_im = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        mask = im > im.mean()
        lbl, n_labs = ndimage.label(mask)
        com = ndimage.measurements.center_of_mass(mask, labels=lbl, index=range(1, n_labs+1))
        return com

    def filter_size(self, im, max_size=1800, min_size=200, sigma=3):
        input_im = im.copy()
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        im = ndimage.gaussian_filter(im, sigma=sigma) #sigma is critical to get right
        mask = im > im.mean()
        label_im, nb_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
        mask_size = (sizes < min_size) | (sizes > max_size)
        remove_pixel = mask_size[label_im]
        remove_pixel.shape
        label_im[remove_pixel] = 0
        input_im[remove_pixel] = 0
        return input_im

    def test_candidates(self, cropped, filtered, x, y, width, height):
#         print('finding centers...')
        coords = self.centers(filtered)
        bcropped = self.crop_corner(x-int(self.dim/2), y-int(self.dim/2), width+self.dim)
#         pretty sure these coords need to be ones from centers()
#         candidates = []
#         for c in coords:
#             candidates.append(self._crop(bcropped, c[1], c[0]))

        candidates = self.crop_ims(bcropped, coords)
        y_pred = np.argmax(self.model.predict(candidates), axis=-1)
        saps = y_pred == 3
        print("n predicted saps: ",np.sum(saps==True))
        coords=np.array(coords)
        return coords[saps]

    def get_saps(self, target, x, y, l, h):
        cropped = self.crop_corner(x, y, l)

#         print('\nsliccing...')
        slc0 = self.slic(cropped, 8, x, y, target)

#         print('filtering...')
        filt0 = self.filter_size(slc0, 90000, 50, 0)

#         print('filtering again...')
        filt1 = self.filter_size(filt0, 90000, 50, 1)
        return self.test_candidates(cropped, filt1, x, y, l, h)

    def slic(self, rgb, n_classes, x, y, target):
        lookupTable, idx,  labels, counts = np.unique(self.samples[1], return_inverse=True, return_counts=True, return_index=True)
        avgs = []
        ims = self.samples[0]
        saps = ims[idx[3]:idx[2]]
        sap_coords = self.coords['sapling']
        for im in saps:
            avgs.append(np.mean(im, axis=(0,1)))
        avgs = np.array(avgs)
        avg = np.mean(avgs, axis=0)
        #  slic it
#         print('segementing...')
        labels1 = seg.slic(rgb[...,:-1], compactness=.00001, n_segments=n_classes,  convert2lab=True, enforce_connectivity=False, start_label=1)
#         print('selecting segment...')
        lookupTable, idx,  labels, counts = np.unique(labels1, return_inverse=True, return_counts=True, return_index=True)
#         print(idx)
# #         print(lookupTable)
#         print(lookupTable)

#         print(labels.shape)
#         print(idx)

        mask = np.zeros(rgb.shape[:2], dtype = "uint8")
        im = np.zeros(rgb.shape[:2], dtype = "uint8")
        diffs = []
        diffs2 = []
        imgs = []
#         print('n segments:', len(lookupTable))
        if len(lookupTable) == 0:
#             cv2.cvtColor(rgb.copy(), cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(rgb.copy(), cv2.COLOR_BGRA2GRAY)
        mask_arr = []
        for (i, segVal) in enumerate(np.unique(labels1)):

            mask = np.zeros(rgb.shape[:2], dtype = "uint8")
            mask[labels1 == segVal] = 255
            mask_arr.append(mask)
            filt2 = cv2.bitwise_and(rgb, rgb, mask = mask)
            mask = filt2 > filt2.mean()

#             plt.imshow(filt2)
#             plt.show()
            label_im, nb_labels = ndimage.label(mask)
            mean_vals = ndimage.sum(filt2, label_im, range(1, nb_labels + 1))
            regions = measure.regionprops(label_im, intensity_image=filt2)
            ims = []
#             print('n regions:',len(regions))
            for i, r in enumerate(regions):
                im = r.intensity_image
#                 inds = np.where(r.intensity_image != [0,0,0,0])
                nonzero_mask = np.all( r.intensity_image != [0, 0, 0, 0], axis=-1)
#                 print(nonzero_mask)
                if np.mean(r.intensity_image[nonzero_mask], axis=0).shape != (4,):
#                     print('BAD!!')
#                     print(np.mean(r.intensity_image[nonzero_mask], axis=0).shape)

#                     plt.imshow(r.intensity_image)
#                     plt.show()
                    continue

                ims.append(np.mean(r.intensity_image[nonzero_mask], axis=0))



#                 print(np.mean(r.intensity_image[r.intensity_image >0], axis=0))

#                 plt.imshow(im)
#                 plt.show()
            ims = np.array(ims)
#             if ims.shape[0] == 1 or len(ims.shape) != 2:
#                 continue
            mean = np.mean(ims, axis=(0))
            diffs.append(np.sum(np.abs(avg -mean)))
#             diffs2.append(np.linalg.norm(avg-mean))
            imgs.append(filt2)
        sdiffs = sorted(diffs)
#         print(diffs)

        simgs = [x for _, x in sorted(zip(diffs, imgs))]
        masks = [x for _, x in sorted(zip(diffs, mask_arr))]

#         print(diffs)
#         print(sdiffs)
        first = sdiffs[sdiffs.index(min(sdiffs))]
        second = sdiffs[sdiffs.index(min(sdiffs))+1]
        ii = 0
#         if second-first < .06:
#             print('segs are close')
#             for mask in masks[:2]:
#                 ii += 1
#                 for ix in range(mask.shape[0]):
#                     for iy in range(mask.shape[1]):
# #                         print(mask[ix, iy])

#                         if mask[ix, iy] == 255:
#                             for c in sap_coords:
#                                 b = False
#                                 if ix == c[0] - x and iy == c[1] - y:
#                                     b = True
#                                     print('1found labeled data in segment', ii)
#                                 if ix == c[1] - x and iy == c[0] - y:
#                                     b = True
#                                     print('2found labeled data in segment', ii)
#                                 if iy == c[0] - x and ix == c[1] - y:
#                                     b = True
#                                     print('3found labeled data in segment', ii)
#                                 if iy == c[1] - x and ix == c[0] - y:
#                                     b = True
#                                     print('4found labeled data in segment', ii)
#                                 if  b:
#                                     print('/t')





#         simgs = [x for _, x in sorted(zip(diffs, imgs))]


#         print(diffs.index(min(diffs)))


#         print('Slic selected segment:')
#         arr = []
#         arr.append(simgs[0])
#         arr.append(simgs[1])
# #         plt.imshow(simgs[sdiffs.index(min(sdiffs))])
#         self.plot(arr)
#         plt.show()
#         print('confidence:', second-first, '\n\n')

        return simgs[0]

#         return imgs[target]

    def blocc(self, im, x,y,l,h, val, thicc=20):
        im[y:y+h,x:x+l] = val
        return im

    def rect(self, im, x,y,l,h, val, thicc=20):
        #   line from x1y1 to x2y1
        im[y:y+thicc,x:x+l] = val

        #   line from x1y1 to x1y2
        im[y:y+h,x:x+thicc] = val

        #   line from x1y2 to x2y2
        im[y+h:y+h+thicc,x:x+l] = val

        #  line from x2y1 to x2y2
        im[y:y+h,x+l:x+l+thicc] = val
        return im

    def get_regions(self):
        bin_im = cv2.cvtColor(self.im.copy(), cv2.COLOR_BGRA2GRAY)
        mask1 = bin_im == 1
        mask2 = bin_im == 0
        mask = np.logical_or(mask1, mask2)

        nim = np.zeros_like(bin_im)
        nim[mask] = .5
        nim[~mask] = 0
        cim = nim.copy()

#         dim = 4000
        inc = 500

        regions = []

        dims = [4000, 2000, 1000, 500]

        for i, d in enumerate(dims):
            for x in range(0, bin_im.shape[0], inc):
                for y in range(0, bin_im.shape[1], inc):
                    print(np.around(((((x+1)*(y+  bin_im.shape[1])*(i+1)))/(bin_im.shape[0]*bin_im.shape[1]*len(dims)))*50, decimals=2)," %      \r", end='')

                    if (np.count_nonzero(cim[x:x+d, y:y+d])/(d*d) < .0001):
                        nim = self.rect(nim, y, x, d, d, 1)
                        cim = self.blocc(cim, y, x, d, d, 1)
                        regions.append([x,y,d])

        plt.imshow(nim)
        plt.show()
        return regions
    
    def binary_im(self, regions):
#             import imageio
        bin_im = cv2.cvtColor(self.im.copy(), cv2.COLOR_BGRA2GRAY)
#         nim = np.zeros_like(bin_im)
        nim = np.zeros_like(bin_im)
        mask1 = bin_im == 1 
        mask2 = bin_im == 0
        mask = np.logical_or(mask1, mask2)
        nim[mask] = .5
        preds = []
        i = 0
        for r in regions:
            pred = self.get_saps(0, r[0], r[1], r[2], r[2])
            for coord in pred:
                x = coord[0] + int((r[0]+r[2]))
                y = coord[1] + int((r[1]+r[2]))

                nim[int(y):int(y+30), int(x):int(x+30)] = 1
            if i % 5 == 0:
                plt.imshow(nim)
                plt.show()
            i += 1
                
            
#                 get_saps(self, target, x, y, l, h):
      
#         for a in ass:
    
#             x1 = a[0][1]
#             y1 = a[0][0]
#             y1 += 1000


#             h = 6000
#             l = 6000

#             x2 = y1+l
#             y2 = x1+h
 
#             h2 = 3000
#             l2 = 4000
      
#             nim = rect(nim, int((y1+h)), int((x1+l)-4000), 4000, 4000, 50)


#             pred = get_saps(2, model, rgb65, data, 32, int((y1+h)), int((x1+l)-4000), 4000, 4000)

#             for coord in pred:
#         #         x = coord[0] + x1
#         #         y = coord[1] + y1
#                 x = coord[0] + int((x1+l)-4000)
#                 y = coord[1] + int((y1+h))

#                 nim[int(x):int(x+50), int(y):int(y+50)] = 1
#         #         nim[int(x), int(y)] = 1
#             break

#         plt.imshow(nim)
#         plt.show()
#         # imageio.imwrite('binary_predictions_bottom30'  + '.tif', nim)
       
