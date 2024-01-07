# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:33:39 2023

@author: Hritik
"""

import cv2 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pyautogui, time

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (48, 48, 3)
model = create_model(input_shape)
model.load_weights('gesture_model.h5')

cap = cv2.VideoCapture(0)
time.sleep(10)
while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'
      continue
    else:
        cv2.imshow('camera input', cv2.flip(image, 1))  
        image = cv2.resize(image, (48,48))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.reshape((1, 48, 48, 3))
        
        prediction = model.predict(image)
        print("Prediction value:", prediction[0][0])
        
        if(prediction[0][0]>=0 and prediction[0][0]<0.33 ):
            pyautogui.keyDown("left")
            pyautogui.keyUp("right")
            pyautogui.keyUp("up")
        elif(prediction[0][0]>=0.33 and prediction[0][0]<0.66):
            pyautogui.keyDown("right")
            pyautogui.keyUp("up")
            pyautogui.keyUp("left")
        else:
            pyautogui.keyDown("up")
            pyautogui.keyUp("left")
            pyautogui.keyUp("right")
    
    
