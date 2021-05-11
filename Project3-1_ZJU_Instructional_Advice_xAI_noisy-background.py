#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Images were retrieved from video lectures of a lecturer in a course
#Then the images were labeled with either unenthusiastic or enthusiastic by an evaluator
#There are 665 images in total

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


# In[2]:


base_dir = r'C:\Users\lawre\OneDrive\Documents\After_WG\Web-Portfolio\Codes\training_dataset1'
train_dir = os.path.join(base_dir, 'training_dataset')

#Directory with training data of identified unenthusiastic teaching images of the lecturer in the course
train_unenthusiastic_dir = os.path.join(train_dir, 'unenthusiastic')

#Directory with training data of identified enthusiastic teaching images of the lecturer in the course
train_enthusiastic_dir = os.path.join(train_dir, 'enthusiastic')


# In[3]:


#Checking names of images
train_unenthusiastic_fnames = os.listdir(train_unenthusiastic_dir)
train_unenthusiastic_fnames.sort()
print(train_unenthusiastic_fnames[:10])

train_enthusiastic_fnames = os.listdir(train_enthusiastic_dir)
print(train_enthusiastic_fnames[:10])

print('total training unenthusiastic images:', len(os.listdir(train_unenthusiastic_dir)))
print('total training enthusiastic images:', len(os.listdir(train_enthusiastic_dir)))


# In[4]:


#Parameters for images; images will be displayed in a 4x4 configuration
nrows = 4
ncols = 4

#Index for iterating over images
pic_index = 0


# In[5]:


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


# In[6]:


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

#Now let's set some hyperparameters for the CNN model which will be trained to be the evaluator!

#Input feature map is 231X231x3: 231X231 for the image pixels, and 3 for
#the three color channels: R, G, and B
img_input = layers.Input(shape=(231, 231, 3))

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

#Flatten feature map to a 1-dim tensor so fully connected layers can be added
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


# In[7]:


#Define the loss function, the optimizer and the metrics
model.compile(loss='binary_crossentropy',    
              optimizer=RMSprop(lr=0.0001),     #tried lr from 0.01 to 0.0001; in this case, lr > 0.001 would give only 0.5 accuracy rate
              metrics=['acc'])

#Plotting the model
from tensorflow.keras.utils import plot_model
import pydot_ng as pydot

plot_model(model, to_file='model_ZJU_noisy-background.png')


# In[8]:


#Let's do data preprocessing!
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
#Flow training images in batches of 23 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,    #This is the source directory for training images
        target_size=(231, 231),    #All images will be resized to 231X231
        classes= ['unenthusiastic', 'enthusiastic'],    #defult: 0 label as index of 0 (i.e. unenthusiastic), vice versa
        batch_size=23,
        #Since using binary_crossentropy loss, I set binary labels
        class_mode='binary')

train_labels = train_generator.classes 
train_labels


# In[9]:


#Now train the CNN model!
history = model.fit_generator(
      train_generator,  
      epochs=6,    #can be increased for achieveing higher acc; 0.9 acc is enough for applying LIME
      )


# In[10]:


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

img = load_img(img_path, target_size=(231, 231))  #this is a PIL image
x = img_to_array(img)  #Numpy array with shape (231, 231, 3)
x = x.reshape((1,) + x.shape)  #Numpy array with shape (1, 231, 231, 3)

#Rescale by 1/255
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


# In[11]:


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


# In[12]:


#Testing the model. However, here I do not have the test data labeled by the evaluator.
#It does not matter since the purpose is to apply LIME on the CNN training model to get the evaluator's tacit knowledge. 
#This is like feature engineeing on instructional enthusiasm in corresponding predictive CNN model!

test_dir = os.path.join(base_dir, 'test_dataset')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,    #did not set (e.g. classes= ['test']) since there is only one folder named "test" with 665 images without labeling unenthusiastic or enthusiastic
        target_size=(231, 231),
        batch_size=23,
        class_mode='binary',
        shuffle=False)    #it is important to specify shuffle=False in order to preserve the order of filenames and predictions.

pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)


#Get classes by np.round
cl = np.round(pred)
#Get filenames (set shuffle=false in generator is important)
filenames=test_generator.filenames

#Data frame
results=pd.DataFrame({"file":filenames,"pr":pred[:,0], "class":cl[:,0]})
results


# In[13]:


print(len(results.loc[results['class'] == 0]))    #class 0 = unenthusiastic
print(len(results.loc[results['class'] == 1]))    #class 1 = enthusiastic


# In[14]:


#All right! Now let's start applying LIME to the CNN model!

#First, Listing the probabilities of being classified as 'enthusiastic' among unenthusiastic images
import skimage

unen_img = []
for i in range(344):
    img_path = os.path.join(train_unenthusiastic_dir, 'unenthusiastic'+' ({0})'.format(i+1)+'.png')
    img = load_img(img_path, target_size=(231, 231))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255
    unen_img.append(x)

len(unen_img)


# In[15]:


prob_of_en_on_unenIMG = []
for i in range(344):
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

#Note: the 344 images below belong to 'unenthusiastic'(labeled by the evaluator), 
#therefore the probabilitie should be close to 0.
#There are some misclassifications (i.e. probabilities are heigher than 0.5)
#Since the model has over 95% accuracy rate, those misclassifications are just a few!


# In[16]:


#Second, Listing the probabilities of being classified as 'enthusiastic' among enthusiastic images
#Note: the 321 images below belong to 'enthusiastic' (labeled by the evaluator), 
#therefore the probabilitie should be close to 1.

en_img = []
for i in range(321):
    img_path = os.path.join(train_enthusiastic_dir, 'enthusiastic'+' ({0})'.format(i+1)+'.png')
    img = load_img(img_path, target_size=(231, 231))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x /= 255
    en_img.append(x)

len(en_img)


# In[17]:


prob_of_en_on_enIMG = []
for i in range(321):
    b = model.predict(en_img[i])
    b = b.flatten()
    b = b.tolist()
    prob_of_en_on_enIMG.append(b)

prob_of_en_on_enIMG


# In[18]:


#https://stackoverflow.com/questions/34990652/why-do-we-need-np-squeeze
#Use numpy.squeeze() function to remove single-dimensional entries (length = 1) from the shape of an array.
#e.g. (1, 3, 3) => (3, 3); (1, 3, 1) = > (3,)
unen_img_squeeze = []    #a list

for i in range(344):
    c = np.squeeze(unen_img[i], axis=0)    #make 3D tensor 2D tensor
    unen_img_squeeze.append(c)

unen_img_squeeze    #a list containing 344 images of 2D tensor


# In[20]:


en_img_squeeze = []

for i in range(321):
    d = np.squeeze(en_img[i], axis=0)
    en_img_squeeze.append(d)

en_img_squeeze


# In[22]:


#Generate segmentation (i.e. superpixel) for image
import skimage.segmentation
import skimage.io
superpixels = skimage.segmentation.quickshift(en_img_squeeze[0], kernel_size=2, max_dist=100, ratio=0.5)
num_superpixels = np.unique(superpixels).shape[0]
skimage.io.imshow(skimage.segmentation.mark_boundaries(en_img_squeeze[0], superpixels))


# In[31]:


print(superpixels.shape)    #231X231
print(np.unique(superpixels))    #154
superpixels    #on en_img_squeeze[0]


# In[32]:


#Generate 665 perturbations (more perturbations, more explanibility power LIME has)
num_perturb = 665
#In each perturbation, 154 num_superpixels would be randomly selected 
#(e.g. in the first perturbation, superpixel #2, #134, #152 are selected; in the second one, #2 and #144 are selected);
#not selected ones would be masked later.
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))  
#What if I select only one of num_superpixels for 154 perturbations? 


# In[34]:


print(perturbations.shape)    #665 perturbations

#Create function to apply perturbations to images
import copy
def perturb_image(img,perturbation,segments): 
  active_pixels = np.where(perturbation == 1)[0]    #a pertubation is an 1D tensor of 0 or 1: (154,); this returns "index" = num_superpixels! 
  mask = np.zeros(segments.shape)    #make all values in superpixels (231X231 matrix) 0
  for active in active_pixels:
      mask[segments == active] = 1    #assign pixels of selected num_superpixels to be 1; mask = 231X231, segments = 231X231, [segments == active] = 231X231 of booleans
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]    #[:,:,1]
  return perturbed_image


# In[37]:


#Show example of perturbations
skimage.io.imshow(perturb_image(en_img_squeeze[0],perturbations[0],superpixels))


# In[50]:


predictions = []     #will contain prediction results of each perturbation
for pert in perturbations:    #try 655 perturbations
  perturbed_img = perturb_image(en_img_squeeze[0],pert,superpixels)
  pred = model.predict(perturbed_img[np.newaxis,:,:,:])    #[1, :,:,:]
  predictions.append(pred)

predictions = np.array(predictions)
print(predictions)
print(predictions.shape)
predictions_sq = np.squeeze(predictions, axis=1)
print(predictions_sq)
print(predictions_sq.shape)


# In[53]:


#Compute distances to original image (i.e. no mask on the image)
import sklearn.metrics
original_image = np.ones(num_superpixels)[np.newaxis,:]    #perturbation with all superpixels enabled (i.e. no mask on the image)
distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()    #different between 665 perturbations and the orinial one 
                                                                                                         #cosine distance is defined as 1.0 minus the cosine similarity.
print(distances.shape)                                                                                   #ravel = reshape(-1, order=order)
print(distances)

#Transform distances to a value between 0 an 1 (weights) using a kernel function
kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function 
#larger the cosine similarity ( = smaller cosine distance values (absolute values) = less 0 values in perturbation or less masked), 
#larger the weight,
#which makes sense since LIME is on "slightly" local changes.
print(weights.shape)
print(weights)


# In[57]:


#Estimate linear model
from sklearn.linear_model import LinearRegression
simpler_model = LinearRegression().fit(X=perturbations, y=predictions_sq, sample_weight=weights)    #sample_weight: array-like of shape (n_samples,). Individual weights for each sample
coeff = simpler_model.coef_
print(coeff.shape)
print(coeff)


# In[72]:


#Use coefficients from linear model to extract top features
num_top_features = 5
top_features = np.argsort(coeff)[0, -num_top_features:] #5 = num_top_features


# In[74]:


#Show only the superpixels corresponding to the top features
mask = np.zeros(num_superpixels) 
mask[top_features] = True #Activate top superpixels to be shown (no masked)
skimage.io.imshow(perturb_image(en_img_squeeze[0],mask,superpixels))


# In[77]:


#USE mark_boundaries
def perturb_image2(mask, segments): 
  active_pixels = top_features
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  return mask

C = perturb_image2(mask, superpixels)
D = np.int64(C)
print(D)
skimage.io.imshow(skimage.segmentation.mark_boundaries(en_img_squeeze[0], D))


# In[91]:


#Use for loop!
num_perturb = 400    #The more, the merrier. I just want to save time here.
num_top_features = 5
predictions_321 = {}
LIME_results_321 = {}
for i in range(321):
    superpixels = skimage.segmentation.quickshift(en_img_squeeze[i], kernel_size=2, max_dist=100, ratio=0.5)    #random each time
    num_superpixels = np.unique(superpixels).shape[0]
    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels)) 
    predictions_321[i] = []     
    for pert in perturbations:    
        perturbed_img = perturb_image(en_img_squeeze[i],pert,superpixels)
        pred = model.predict(perturbed_img[np.newaxis,:,:,:])   
        predictions_321[i].append(pred)
    predictions_321[i] = np.array(predictions_321[i]) 
    predictions_321[i] = np.squeeze(predictions_321[i], axis=1)
    original_image = np.ones(num_superpixels)[np.newaxis,:] 
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) 
    simpler_model = LinearRegression().fit(X=perturbations, y=predictions_321[i], sample_weight=weights)  
    coeff = simpler_model.coef_
    top_features = np.argsort(coeff)[0, -num_top_features:]
    def perturb_image2(mask, segments): 
        active_pixels = top_features
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1 
        return mask
    C = perturb_image2(mask, superpixels)
    D = np.int64(C)
    LIME_results_321[i] = skimage.segmentation.mark_boundaries(en_img_squeeze[i], D)


# In[94]:


skimage.io.imshow(LIME_results_321[320])


# In[99]:


from matplotlib.pyplot import show    #the show() function in matplotlib is used to display all figures

images = []
for i in range(321):
    images.append(LIME_results_321[i])


for i in images:
    skimage.io.imshow(i)
    show()
    time.sleep(0.01)


# In[ ]:


#Hurah! I built LIME by my own! 


# In[106]:


#Now let's try using the LIME package 
from lime import lime_image
from skimage.segmentation import mark_boundaries
import time
tmp = time.time()
print(time.time() - tmp)

explainer = lime_image.LimeImageExplainer()
for i in range(321): 
    explanation = explainer.explain_instance(en_img_squeeze[i], model, hide_color=0)
    temp, mask = explanation.get_image_and_mask(0, positive_only=False, num_features=5, hide_rest=False)
    plt.imshow(mark_boundaries(temp, mask))
    show()
    time.sleep(0.01)

