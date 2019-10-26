# based on code from https://www.tensorflow.org/tutorials


# The program should finish execution within 6 minutes. Irrespective of seed value, the test errors are very close. So focus on modifying your code to get best results on 
# the current training set (made from any one seed value)

import tensorflow as tf
import numpy as np

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# different seed value gives different training sets. Make sure your algorithm works well irrespective of the training set.
# specify path to training data and testing data

folderbig = "big"
foldersmall = "small"

train_x_location = foldersmall + "/" + "x_train.csv"
train_y_location = foldersmall + "/" + "y_train.csv"
test_x_location = folderbig + "/" + "x_test.csv"
test_y_location = folderbig + "/" + "y_test.csv"

print("Reading training data")
# each image is stored as a row
x_train_2d = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")             # reading training images as 2D. One row = 1 image
x_train_3d = x_train_2d.reshape(-1,28,28,1)         # here each image is stored as a 28x28x1 array. Only 1 band => gray level images
x_train = x_train_3d
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

print("Pre processing x of training data")
x_train = x_train / 255.0              # scaling to the range : 0-255


#*****************************************************************************

# defining the training model
# try changing number of filters, layers, dropouts.
# Since training data set is much smaller, avoiding over-fitting is of utmost importance. Regularization

model = tf.keras.models.Sequential([    # sequential model => VGG
    tf.keras.layers.MaxPool2D(4, 4, input_shape=(28,28,1)), # Don't change. Shrinking training images by 1/4 from 28x28 to 7x7, input dimension should be specified in the first layer
	
	# tf.keras.layers.Conv2D(64,(3,3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
	tf.keras.layers.Conv2D(128,(3,3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.MaxPool2D(2, 2),          # shrink image by 1/4
	
	tf.keras.layers.Conv2D(256,(3,3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
	# tf.keras.layers.Conv2D(128,(3,3), padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
	tf.keras.layers.MaxPool2D(2, 2), 
	
    tf.keras.layers.Flatten(),               # image needs to be flattened (converting from 3D TO 1D) before Dense layer
	
	# Dense layer => Fully Connected Layer. Here 512 nodes + ReLU
   
	# tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
	tf.keras.layers.Dense(1024, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)), 
	
						
	# use bias-regularizer too to tune the bias weights too. Regularization parameter = 0.001 (No use)
	tf.keras.layers.Dropout(0.45),  # 0.2 => drop 20% of nodes (for regularization or avoiding over-fitting) ********************[0.45]
    # last layer is softmax
	
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)   # 10 output labels (Don't change). Gives out a probability vector as required
])


# not using regularization gave higher accuracy somehow.
# adding a FC layer improved accuracy




# Our model is fully created. Now, we just have to optimize it.
# loss='categorical_entropy' expects input to be one-hot encoded
# loss='sparse_categorical_entropy' expects input to be the category as a number
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # input not given as one-hot. This label will convert input labels to one-hot and then calculate the cross-entropy
              metrics=['accuracy'])






#**********************************************************

print("train")
# model.fit(x_train, y_train, epochs=7)  
model.fit(x_train, y_train, epochs=50, batch_size=300) # Running the model. Use more epochs (iterations for tuning weights) in project. (Significant improvement on high epoch value)
# default batch size (stochastic gradient descent) is 32,we can explicitly specify the value. To fine tune, change this.


# Increasing epoch gives the best return !!!! 

# 0.8965



print("Reading testing data")  # reading the testing data
x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
x_test_3d = x_test_2d.reshape(-1,28,28,1)
x_test = x_test_3d
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

print("Pre-processing testing data") # pre-processing testing data
x_test = x_test / 255.0

print("Evaluating model on the testing data")  # evaluating our model
results = model.evaluate(x_test, y_test)
print('test loss, test accuracy:', results)

