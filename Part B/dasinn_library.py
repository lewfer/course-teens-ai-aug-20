# Functions to help with AI course


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

# ------------------------- For Tensorflow Playground exercise ----------------------------------

# Define the range of values for x1 and x2
minx = -6
maxx = 6
midx = 0

# Generate a set to random "things" (humans or aliens) 
def randThing(fromval, toval, size, cols):
    return np.random.randint(fromval*10,toval*10,size=(size, cols)) /10.0

# Create a random data set for training and testing
# We don't have access to the Tensorflow data sets, so let's fake them using some code.  This code uses Numpy, which is a Python library for fast manipulation of arrays of numbers.
def createDataSet1(count):
    humans = pd.DataFrame(randThing(minx,midx,count,2), columns=list(['x1','x2']))
    humans["human"] = 0
    aliens = pd.DataFrame(randThing(midx,maxx,count,2), columns=list(['x1','x2']))
    aliens["human"] = 1
    all = pd.concat([humans,aliens])
    return all    

# Create a random data set for training and testing
def createDataSet2(count):
    half= int(count/2
             )
    humans1 = pd.DataFrame(randThing(minx,midx,half, 2), columns=list(['x1','x2']))
    humans1["human"] = 1
    humans2 = pd.DataFrame(randThing(midx,maxx,half, 2), columns=list(['x1','x2']))
    humans2["human"] = 1

    aliens1 = pd.DataFrame(np.column_stack((randThing(minx,midx,half, 1), randThing(midx,maxx,half, 1))), columns=list(['x1','x2']))
    aliens1["human"] = 0
    aliens2 = pd.DataFrame(np.column_stack((randThing(midx,maxx,half, 1), randThing(minx,midx,half, 1))), columns=list(['x1','x2']))
    aliens2["human"] = 0
    all = pd.concat([humans1,aliens1,humans2,aliens2])
    return all

# ------------------------- For Simple image exercise ----------------------------------

# Generate numRandom random images and numTargets target images
def generateRandomImages(imsize, numRandom, targets, numTargets):
    # Flatten targets
    flatTargets = []
    for target in targets:
        flatTargets.append(list(np.reshape(target, imsize*imsize)))
    flatTargets = np.array(flatTargets)
    
    # Generate some random images in X, specifying if they match the target in y
    X = []
    y = []
    for i in range(numRandom):
        image = list(np.random.randint(0,2,size=imsize*imsize)) # generate random bits
        X.append(image)                                         # add image to list
        y.append(any((image == x).all() for x in flatTargets))      # add True or False to say if image is in our targets

    # Add some of the target images in
    for i in range(numTargets):
        image = flatTargets[random.randint(0,len(flatTargets)-1)]
        X.append(image)
        y.append(True)

    X = np.array(X)
    y = np.array(y)
    
    return X, y

def sigmoid(score):
    return 1/(1 + np.exp(-score))

def predictImage(test, weights, bias):
    #print("bias",bias)
    sum = bias
    for i,weight in enumerate(weights):
        #print(weight)
        sum += test[i] * weight[0]
    #print("sum",sum)
    return sigmoid(sum) #> layer2weights[0]

def displayImages(X, y, predictions, imsize):
    # Display the images
    num_of_samples = len(X)

    cols = min(5, len(X))
    rows = int(round(num_of_samples / cols) )
    
    # Prepare a grid
    fig, axs = plt.subplots(nrows=rows, ncols = cols, figsize=(10,1*rows), squeeze=False)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=0.4)

    # Cycle through the images (i) and digits (j)
    for i in range(cols):
        for j in range(rows):
            idx = j*cols+i
            im = np.invert(X[idx]).reshape(imsize,imsize)
            axs[j][i].imshow(im, cmap=plt.get_cmap("gray"))
            #axs[j][i].axis("off")
            if y is None:
                axs[j][i].set_title(str(idx))
            elif predictions is None:
                axs[j][i].set_title(str(int(y[idx])))
            else:
                correct = y[idx] == predictions[idx]
                #axs[j][i].set_title(str(idx) + " " + str(y[idx]) + " " + str(predictions[idx]))
                axs[j][i].set_title("A:" + str(int(y[idx])) + " P:" + str(predictions[idx][0]) + ("" if correct else " WRONG"))
            # Turn off tick labels
            axs[j][i].set_yticklabels([])
            axs[j][i].set_xticklabels([])
            axs[j][i].tick_params(axis=u'both', which=u'both',length=0)


# ------------------------- For colour images exercise ----------------------------------


def plot_image(i, true_label, img, class_names):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[true_label[i][0]])
    
def plot_value_array(i, predictions_array, true_label, class_names):
  predictions_array, true_label = predictions_array[i], true_label[i][0]
  plt.grid(False)
  plt.xticks(rotation=90)
  plt.yticks([])
  thisplot = plt.bar(class_names, predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')
  #plt.xlabel(class_names)    
    
def plotImageResults(num_rows, num_cols, predictions, test_x, test_y, class_names):
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plot_image(i, test_y, test_x, class_names)
      plt.subplot(num_rows, 2*num_cols, 2*i+2)
      plot_value_array(i, predictions, test_y, class_names)
    plt.show()    
    
def plotImages(num_rows, num_cols, x, y, class_names):
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*num_cols, 2*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, num_cols, i+1)
      plot_image(i, y, x, class_names)
    plt.show()        

# ------------------------- For fashion images exercise ----------------------------------

def plot_fashion_image(i, predictions_array, true_label, img, class_names):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_fashion_value_array(i, predictions_array, true_label, class_names):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(rotation=90)
  plt.yticks([])
  thisplot = plt.bar(class_names, predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')
  #plt.xlabel(class_names)

def plot_fashion_image_results(num_rows, num_cols, predictions, test_x, test_y, class_names):
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_fashion_image(i, predictions, test_y, test_x, class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_fashion_value_array(i, predictions, test_y, class_names)
    plt.show()

def plot_fashion_images(num_rows, num_cols, x, y, class_names):
    # Plot the first few images, with labels
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y[i]])
    plt.show()

