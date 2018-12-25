# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/hist.png "Hist"
[image2]: ./imgs/center.jpg "Grayscaling"
[image3]: ./imgs/side1.jpg "Recovery Image"
[image4]: ./imgs/side2.jpg "Recovery Image"
[image5]: ./imgs/side3.jpg "Recovery Image"
[image6]: ./imgs/center.jpg "Normal Image"
[image7]: ./imgs/flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 filters and two 3x3 filters and depths between 24 and 64 (model.py lines 33-47) 

The model includes RELU layers to introduce nonlinearity (code line 36), and the data is normalized in the model using a Keras lambda layer (code line 34). 

#### 2. Attempts to reduce overfitting in the model

dropout layers in my build performed not really good, so I didn't use it . 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 50). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 49).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , and collected more data on every corner. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test some model and choose one.

My first step was to use a convolution neural network model similar to the NVIDIA's self-driving network, I thought this model might be appropriate because it was proved work in their development.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I try to use dropout ,but it not really work.

Then I try to collect more data

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track on the corner, to improve the driving behavior in these cases, I try
to collet more data from these place.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 33-47) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						|
| Normlayer             | Normlization                                  |
| Croplayer             | output 320x90x3 RGB image                     |
| Convolution 5x5x24    | 2x2 stride, valid padding ,relu             	|
| convolution 5x5x36	| 2x2 stride, valid padding ,relu				|
| convolution 5x5x48	| 2x2 stride, valid padding ,relu				|
| convolution 3x3x64	| 1x1 stride, valid padding ,relu				|
| convolution 3x3x64	| 1x1 stride, valid padding ,relu				|
| Fully connected		| 1000          							    |
| Fully connected       | 200                                           |
| Fully connected       | 100                                           |
| Fully connected 		| 50        									|
| Fully connected 		| 1        								     	|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

This is the plot of images distribution at the first collection,really unbalanced,so I decided to collect more data on the corner:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the track center, These images show what a recovery looks like starting from side of track :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 19947 number of data points. I then preprocessed this data by flip them.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by validation loss, I used an adam optimizer so that manually training the learning rate wasn't necessary.# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./imgs/center.jpg "Grayscaling"
[image3]: ./imgs/side1.jpg "Recovery Image"
[image4]: ./imgs/side2.jpg "Recovery Image"
[image5]: ./imgs/side3.jpg "Recovery Image"
[image6]: ./imgs/center.jpg "Normal Image"
[image7]: ./imgs/flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 filters and two 3x3 filters and depths between 24 and 64 (model.py lines 33-47) 

The model includes RELU layers to introduce nonlinearity (code line 36), and the data is normalized in the model using a Keras lambda layer (code line 34). 

#### 2. Attempts to reduce overfitting in the model

dropout layers in my build performed not really good, so I didn't use it . 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 50). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 49).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , and collected more data on every corner. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test some model and choose one.

My first step was to use a convolution neural network model similar to the NVIDIA's self-driving network, I thought this model might be appropriate because it was proved work in their development.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I try to use dropout ,but it not really work.

Then I try to collect more data

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track on the corner, to improve the driving behavior in these cases, I try
to collet more data from these place.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 33-47) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   						|
| Normlayer             | Normlization                                  |
| Croplayer             | output 320x90x3 RGB image                     |
| Convolution 5x5x24    | 2x2 stride, valid padding ,relu             	|
| convolution 5x5x36	| 2x2 stride, valid padding ,relu				|
| convolution 5x5x48	| 2x2 stride, valid padding ,relu				|
| convolution 3x3x64	| 1x1 stride, valid padding ,relu				|
| convolution 3x3x64	| 1x1 stride, valid padding ,relu				|
| Fully connected		| 1000          							    |
| Fully connected       | 200                                           |
| Fully connected       | 100                                           |
| Fully connected 		| 50        									|
| Fully connected 		| 1        								     	|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the track center, These images show what a recovery looks like starting from side of track :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 19947 number of data points. I then preprocessed this data by flip them.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by validation loss, I used an adam optimizer so that manually training the learning rate wasn't necessary.
