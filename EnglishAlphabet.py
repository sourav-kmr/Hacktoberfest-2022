#Import the libraries
import cv2 as cv
import numpy as np


# Read data and use converters
# to convert the alphabets to
# Numeric value.
data= np.loadtxt('letter-recognition',
				dtype= 'float32',
				delimiter = ',',
				converters= {0: lambda ch: ord(ch)-ord('A')})

# split the data into train_data
# and test_data
train_data, test_data = np.vsplit(data,2)

# split train_data and test_data
# to features and responses.
responses, training = np.hsplit(train_data,[1])
classes, testing = np.hsplit(test_data,[1])

# Create the knn classifier
knn = cv.ml.KNearest_create()
knn.train(training, cv.ml.ROW_SAMPLE, responses)

# Obtain the results of the classifier
# determine the number of neighbors.
ret, Output, neighbours,
distance = knn.findNearest(testing, k=7)

# Match the Output to find the
# number of wrong predictions.
correct_OP = np.count_nonzero(Output == classes)

#calculate accuracy and display it.
accuracy = (correct_OP*100.0)/(10000)
print( accuracy )
