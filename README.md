# HANDSIGN
For this project, we tried to detect the hand sign of the user by using mediapipe from Google and using a machine learning model to predict the number from those hand signs and then use it as a password to unlock the door.
# Dataset
In this project, we created a dataset by ourselves by taking a photo of our right-hand sign in the different locations for each number 0–9.

For the number ‘0’, we used our fist as the number 0 and we took 15 photos for this hand sign.

For the number ‘1’, we used the sign shown below as the number 1 and we took 40 photos for this hand sign.

For the number ‘2’, we used the sign shown below as the number 2 and we took 43 photos for this hand sign.

For the number ‘3’, we used the sign shown below as the number 3 and we took 26 photos for this hand sign.

For the number ‘4’, we used the sign shown below as the number 4 and we took 21 photos for this hand sign.

For the number ‘5’, we used the sign shown below as the number 5 and we took 27 photos for this hand sign.

For the number ‘6’, we used the sign thumbs up as shown below as the number 6 and we took 28 photos for this hand sign.

For the number ‘7’, we used the sign shown below as the number 7 and we took 27 photos for this hand sign.

For the number ‘8’, we used the sign shown below as the number 8 and we took 28 photos for this hand sign.

For the number ‘9’, we used the sign shown below as the number 9 and we took 27 photos for this hand sign.

However, during we took a photo to create a dataset, some photos that we provided can not be detected by mediapipe. Therefore, we need to delete those photos to make our model can work.
Because of this reason, The total number of images that we used to create the dataset of hand sign numbers have 282 images.
# METHOD
For the preprocessing part, we pass our hand-sign image dataset into a mediapipe model, the mediapipe model will detect our hand and show the hand landmark for our hand in the image. Mediapipe can interpret our hand landmark as a normalized coordinate.

Then we use those coordinates as a representative of our hand sign image dataset in a CSV file format. Then we put those coordinates as an input for the machine learning model training.

We choose the neural network model for our main model. The model is consist of 1 input layer with 63 input mode, 5 hidden layers with 64,128,256,128, and 64 nodes respectively and 1 output layer with 10 nodes and use the softmax activation function to obtain the probability of the 10 possible class (number 0–9).

We can interpret our hand-sign as a number by passing our hand into the mediapipe model, extracting the normalized coordinate of our hand from the mediapipe model, and then passing the coordinate to our trained model, the model will predict the class (number 0–9) based on the input coordinate we pass on.
# HARDWARE
- Raspberry pi 4 model B

- Coral edge TPU

- Webcam camera

- Relay control (5V to 12V)

- Power supply (12 V)

- Solenoid lock (12 V)

# TESTING SYSTEM
After finish build hardware, upload all code and model in the raspberry pi 4 model B. We test our system with true password and wrong password by using the steps below.

1. Put your hand in front of the webcam camera and do the sign number

2. Wait for the system receives the result for each digit of the password

3. The system will interpret the hand sign into the number and show it on the screen

4. Do the second and third steps 3 times (we need 4 digits of the password)

5. If your password is correct the system will send the signal to unlock the solenoid lock, otherwise, the door will be locked

# Result
- For checking the result, we tried to split the dataset into training data 80% and testing data 20% of the total dataset. Then we tried to create the confusion matrix and find the accuracy of the model and the results are in confusion_matrix.png, classicfication_report.png.
- From our model, the accuracy of the system was 100%. So we can convert the user’s hand sign into the number and use it to unlock the solenoid lock and increase the safety of the house and make the user more comfortable.

# Authors
Karn Kiattikunrat 6414553344 karn.kia@ku.th

Yanatorn Chadavadh 6414553361 yanatorn.ch@ku.th

Panupong Sunkom 6414553395 panupong.sunk@ku.th

This article is part of ICT730 Hardware Designs for Ai & IoT TAIST-Tokyo Tech program
