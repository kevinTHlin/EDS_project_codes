#!/usr/bin/env python
# coding: utf-8

# In[19]:


#Images were retrieved from video lectures of a lecturer in a course with a simpler background 
#Then the images were labeled with either unenthusiastic or enthusiastic by an evaluator
#There are 73 images in total

import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import math 
import datetime
import time
import os


# In[20]:


base_dir = r'C:\Users\lawre\OneDrive\Documents\After_WG\Web-Portfolio\Codes\training_dataset2'
train_dir = os.path.join(base_dir, 'training_dataset')

#Directory with training data of identified unenthusiastic teaching images of the lecturer in the course
train_unenthusiastic_dir = os.path.join(train_dir, 'unenthusiastic')

#Directory with training data of identified enthusiastic teaching images of the lecturer in the course
train_enthusiastic_dir = os.path.join(train_dir, 'enthusiastic')


# In[21]:


#Checking names of images
train_unenthusiastic_fnames = os.listdir(train_unenthusiastic_dir)
train_unenthusiastic_fnames.sort()
print(train_unenthusiastic_fnames[:10])

train_enthusiastic_fnames = os.listdir(train_enthusiastic_dir)
print(train_enthusiastic_fnames[:10])

print('total training unenthusiastic images:', len(os.listdir(train_unenthusiastic_dir)))
print('total training enthusiastic images:', len(os.listdir(train_enthusiastic_dir)))


# In[22]:


#Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

#Index for iterating over images
pic_index = 0


# In[23]:


#Seting up matplotlib fig, and sizing it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_unenthusiastic_pix = [os.path.join(train_unenthusiastic_dir, fname) 
                for fname in train_unenthusiastic_fnames[pic_index-8:pic_index]]
next_enthusiastic_pix = [os.path.join(train_enthusiastic_dir, fname) 
                for fname in train_enthusiastic_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_unenthusiastic_pix+next_enthusiastic_pix):
  #Setting up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')    #No showing axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# In[24]:


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

#Now let's set some hyperparameters for the CNN model which will be trained to be the evaluator!

#Input feature map is 275X275x3: 275X275 for the image pixels, and 3 for
#the three color channels: R, G, and B
img_input = layers.Input(shape=(275, 275, 3))

#First convolution extracts 16 filters that are 3x3
#Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

#Second convolution extracts 32 filters that are 3x3
#Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

#Third convolution extracts 64 filters that are 3x3
#Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

#Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

#Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

#Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

#Create model:
#input = input feature map
#output = input feature map + stacked convolution/maxpooling layers + fully 
#connected layer + sigmoid output layer
model = Model(img_input, output)
model.summary()


# In[25]:


#Define the loss function, the optimizer and the metrics
model.compile(loss='binary_crossentropy',    
              optimizer=RMSprop(lr=0.00001),     
              metrics=['acc'])

#Plotting the model
from tensorflow.keras.utils import plot_model
import pydot_ng as pydot

plot_model(model, to_file='model_SOU_simple-background.png')


# In[26]:


#Let's do data preprocessing!
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
#Flow training images in batches of 6 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  #This is the source directory for training images
        target_size=(275, 275),  #All images will be resized to 275X275
        classes= ['unenthusiastic', 'enthusiastic'],    #defult: 0 label as index of 0 (i.e. unenthusiastic), vice versa
        batch_size=5,
        #Since using binary_crossentropy loss, I set binary labels
        class_mode='binary')

train_labels = train_generator.classes 
train_labels


# In[27]:


#Now train the CNN model!
history = model.fit_generator(
      train_generator,  
      epochs=15    
      )


# In[28]:


#Visualing feature map!

import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

#Let's define a new Model that will take an image as input, and will output
#intermediate representations for all layers in the previous model after the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

#Let's prepare a random input image of a unenthusiastic or ununenthusiastic from the training set.
unenthusiastic_img_files = [os.path.join(train_unenthusiastic_dir, f) for f in train_unenthusiastic_fnames]
enthusiastic_img_files = [os.path.join(train_enthusiastic_dir, f) for f in train_enthusiastic_fnames]
img_path = random.choice(unenthusiastic_img_files + unenthusiastic_img_files)

img = load_img(img_path, target_size=(275, 275))  #this is a PIL image
x = img_to_array(img)  #Numpy array with shape (275, 275, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 275, 275, 3)

# Rescale by 1/255
x /= 255

#Let's run our image through our network, thus obtaining all
#intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

#These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]    #NOTE: it is model.layers[1:] instead of model.layers!

#Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    #Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]    #number of features in feature map
    #The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    #We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      #Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      #We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    #Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')


# In[29]:


#Retrieve a list of accuracy results on training and validation data
#sets for each training epoch
acc = history.history['acc']

#Retrieve a list of list results on training and validation data
#sets for each training epoch
loss = history.history['loss']

#Get number of epochs
epochs = range(1, len(acc)+1, 1)

#Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.title('Training accuracy')

plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.title('Training loss')


# In[30]:


#All right! Now let's start applying LIME to the CNN model!

#First, Listing the probabilities of being classified as 'enthusiastic' among unenthusiastic images
import skimage

unen_img = []
for i in range(31):
    img_path = os.path.join(train_unenthusiastic_dir, 'unenthusiastic'+str( (i+1))+'.png')
    img = load_img(img_path, target_size=(275, 275))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255
    unen_img.append(x)

len(unen_img)


# In[31]:


prob_of_en_on_unenIMG = []
for i in range(31):
    a = model.predict(unen_img[i])
    a = a.flatten()
    a = a.tolist()
    prob_of_en_on_unenIMG.append(a)

prob_of_en_on_unenIMG
#Get predicted class probabilities (binary classification) 
#the dense layer consisting of one unit with an activation function of the sigmoid. 
#sigmoid function outputs a value in the range [0,1] which corresponds to 
#the probability of the given sample belonging to a positive class (i.e. class 1).
#Recall: in ImageDataGenerator, classes= ['x', 'y']
#where the defult is: 0 label as negative class as 'x', vice versa!

#Note: the 31 images below belong to 'unenthusiastic'(labeled by the evaluator), 
#therefore the probabilitie should be close to 0.
#There are some misclassifications (i.e. probabilities are heigher than 0.5)
#Since the model has over 95% accuracy rate, those misclassifications are just a few!


# In[32]:


#Second, Listing the probabilities of being classified as 'enthusiastic' among enthusiastic images
#Note: the 321 images below belong to 'enthusiastic' (labeled by the evaluator), 
#therefore the probabilitie should be close to 1.

en_img = []
for i in range(42):
    img_path = os.path.join(train_enthusiastic_dir, 'enthusiastic'+str( (i+1))+'.png')
    img = load_img(img_path, target_size=(275, 275))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255
    en_img.append(x)

len(en_img)


# In[33]:


prob_of_en_on_enIMG = []
for i in range(42):
    b = model.predict(en_img[i])
    b = b.flatten()
    b = b.tolist()
    prob_of_en_on_enIMG.append(b)

prob_of_en_on_enIMG


# In[34]:


#https://stackoverflow.com/questions/34990652/why-do-we-need-np-squeeze
#Use numpy.squeeze() function to remove single-dimensional entries (length = 1) from the shape of an array.
#e.g. (1, 3, 3) => (3, 3); (1, 3, 1) = > (3,)
unen_img_squeeze = []    #a list

for i in range(31):
    c = np.squeeze(unen_img[i], axis=0)    #make 3D tensor 2D tensor
    unen_img_squeeze.append(c)


# In[35]:


en_img_squeeze = []

for i in range(42):
    d = np.squeeze(en_img[i], axis=0)
    en_img_squeeze.append(d)


# In[36]:


import skimage.segmentation
import skimage.io
import copy
import sklearn
from sklearn.linear_model import LinearRegression

#Create function to apply perturbations to images
def perturb_image(img,perturbation,segments): 
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image

num_perturb = 200    #The more, the merrier. I just want to save time here.
num_top_features = 5
predictions_42 = {}
LIME_results_42 = {}
for i in range(42):
    superpixels = skimage.segmentation.quickshift(en_img_squeeze[i], kernel_size=2, max_dist=100, ratio=0.5)    #random each time
    num_superpixels = np.unique(superpixels).shape[0]
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels)) 
    predictions_42[i] = []     
    for pert in perturbations:    
        perturbed_img = perturb_image(en_img_squeeze[i],pert,superpixels)
        pred = model.predict(perturbed_img[np.newaxis,:,:,:])   
        predictions_42[i].append(pred)
    predictions_42[i] = np.array(predictions_42[i]) 
    predictions_42[i] = np.squeeze(predictions_42[i], axis=1)
    original_image = np.ones(num_superpixels)[np.newaxis,:] 
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) 
    simpler_model = LinearRegression().fit(X=perturbations, y=predictions_42[i], sample_weight=weights)  
    coeff = simpler_model.coef_
    top_features = np.argsort(coeff)[0, -num_top_features:]
    def perturb_image2(mask, segments): 
        active_pixels = top_features
        for active in active_pixels:
            mask[segments == active] = 1 
        return mask
    mask =  np.zeros(superpixels.shape)
    C = perturb_image2(mask, superpixels)
    D = np.int64(C)
    LIME_results_42[i] = skimage.segmentation.mark_boundaries(en_img_squeeze[i], D)


# In[37]:


from matplotlib.pyplot import show    #the show() function in matplotlib is used to display all figures

images = []
for i in range(42):
    images.append(LIME_results_42[i])

for i in images:
    skimage.io.imshow(i)
    show()
    time.sleep(0.01)

