#this first library that is installed if numpy
#the second one is matplotlib
#next is tensorflow
#Convolutional Neural Network is used to classify image or audio data
#then we installed opencv-python :its work : it is used for classifying image and performing computer vision tasks :
# such as object tracking ,landmark detection and many more
from cgi import test
import imp
from operator import mod
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras import Sequential
from keras import models,layers
(training_images,training_labels),(testing_images,testing_labels)=cifar10.load_data()
training_images,testing_images=training_images/255,testing_images/255
#the pixel values of each image is reduced from 255 to 1 by the above step
class_names=['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
#the images is not be of high resolution
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()
training_images=training_images[:20000]
training_labels=training_labels[:20000]
testing_images=testing_images[:4000]
testing_labels=testing_labels[:4000]
#this is done just to reduce the time required to train the network
'''
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
#convolutional layer finds features that are present in a particular picture
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))
#epocs means how many itmes the model is going to see the same data over and over again
loss,accuracy=model.evaluate(testing_images,testing_labels)
print(f"Loss:{loss}")
print(f"accuracy:{accuracy}")
model.save("image_classifier.model")
'''
#the commented code needs to be run once after that it gets stored in the models folder and then can be used by the following commands
model=models.load_model("image_classifier.model")
img=cv.imread('horse.jpg')
#in open cv we load the pic with bgr colour scheme however all the pic we trained the model on used rgb colour scheme
img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(img,cmap=plt.cm.binary)
prediction=model.predict(np.array([img])/255)
index=np.argmax(prediction)
#argmax retruns the max value of prediciton 
print(f"Prediction is {class_names[index]}")
plt.show()