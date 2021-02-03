#!/usr/bin/env python
# coding: utf-8

# # All Imports

# In[1]:


import tensorflow
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


# # Pre Processing

# * Step1- Load Dataset

# In[2]:


def load_dataset():
    #load dataset
    (trainX,trainY),(testX,testY) = cifar10.load_data()
    # don't need to reshape this as already in the shape of  (50000 X 32 X 32 X 3)
    
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    return trainX, trainY, testX, testY
    


# * Step2- Prepare Pixels

# In[3]:


def prep_pixels(train,test):
    #convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    
    # return normalized images
    return train_norm, test_norm
    
    


# # Define Model

# In[10]:


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))#conv1a
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))#conv1b
    model.add(MaxPooling2D((2, 2)))
    
    #hidden layer2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))#conv2a
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))#conv2a
    model.add(MaxPooling2D((2, 2)))
    
    #hidden layer3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))#conv3a
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))#conv3b
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(model.summary())
    return model


# # Run and Save the Model

# In[25]:


def run_save_model():
    #load dataset
    trainX, trainY, testX, testY = load_dataset()

    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)

    # define model
    model = define_model()

    #fit the model
    model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=2)

    #save the model to disk
    model.save('final_model.h5')
    
    #evaluate
    loss, accuracy = model.evaluate(testX, testY, verbose=0)
    print("Accuracy of the model is ",round((accuracy * 100.0),2))
    
  
    


# In[26]:


#run the model
run_save_model()


# # Loading the saved model from the disk and finding accuracy

# In[29]:


model = tensorflow.keras.models.load_model('final_model.h5')
#evaluate the model
trainX, trainY, testX, testY = load_dataset()
loss, accuracy = model.evaluate(testX, testY, verbose=0)
print("Accuracy of the model is ",round((accuracy * 100.0),2))


# * Thus the accuracy of the model is **70.73%**
