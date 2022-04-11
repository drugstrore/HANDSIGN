import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
from tflite_runtime.interpreter import Interpreter 
import numpy as np
import time
import pycoral.utils.edgetpu 
pycoral.utils.edgetpu.load_edgetpu_delegate() 

########### SET UP PIN ##########
doorlock_pin = 21
GPIO.setmode(GPIO.BCM)
GPIO.setup(doorlock_pin,GPIO.OUT)
GPIO.output(doorlock_pin,GPIO.LOW)
###########  (END) SET UP PIN ##########

########### Load the model ############
new_model = pycoral.utils.edgetpu.make_interpreter('/home/pi/Desktop/model.tflite') ###
new_model.allocate_tensors()
input_details = new_model.get_input_details()
output_details = new_model.get_output_details()
input_shape = input_details[0]['shape']
#######################################

########### Hand landmark detection ########
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

sec = 12 #set index to check the same prediction result 
check = [] #receive prediction result
password = [] #receive complete prediction result
txt = 'PUT YOUR HAND ON THE CAMERA'
True_password = '6047' #set the password for this system

######## Receive image from webcam and use mediapipe to find hand landmark ######
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    results = hands.process(image) 

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    ######## Obtain the hand landmark ##############
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        coordinates = []
        for k in range(0, 21):
            coordinates.append(results.multi_hand_landmarks[0].landmark[k].x)
            coordinates.append(results.multi_hand_landmarks[0].landmark[k].y)
            coordinates.append(results.multi_hand_landmarks[0].landmark[k].z)
        coordinates = np.array(coordinates)
        
        ###### Use coordinate as input data to the model #######
        input_data = coordinates.reshape(-1,1).T
        input_data = np.float32(input_data)
        new_model.set_tensor(input_details[0]['index'], input_data)
        new_model.invoke()
        
        ######## Get the prediction result ##########
        result = new_model.get_tensor(output_details[0]['index'])
        result = np.argmax(result)
        print(result)
        
        ######## Compare the presiction result with the true password #######
        check.append(result)
        k = 0
        if len(check) > sec:
            #print(B)
            for i in range(2,sec+2):
                if check[-i] == check[-i+1]:
                    k += 1
                    #print(k)
                    if k == sec:
                        password.append(check[-1])
                        txt2 = 'Your Password Is : '
                        pass2txt = ''.join(str(E) for E in password)
                        txt = txt2+pass2txt
                        check = []
                        k = 0
                else:
                    k = 0
        ####### After obtain 4 number, check the password is correct or not then send signal to relay #####           
        if len(password) == 4 and len(check) >6:
            if txt == 'Your Password Is : ' + True_password:
                txt = 'Password Correct, Welcome Home!'
                password = []
                check = []
                signal = 1
                GPIO.output(doorlock_pin,GPIO.HIGH)
            else:
                txt = 'Password Incorrect, Please Try Again'
                password = []
                check = []
                signal = 0
                GPIO.output(doorlock_pin,GPIO.LOW)
        
    ####### Create user interface ###########
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    text_size, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,1,3)
    text_w,text_h = text_size
    cv2.rectangle(image,(20-10,20-10),(20+text_w+10,20+text_h+10),(0,0,0),-1)
    cv2.putText(image, txt, (20, 20+text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3)
    

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
