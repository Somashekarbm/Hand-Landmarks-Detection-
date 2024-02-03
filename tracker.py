import cv2 as cv
import time 
import mediapipe as mp

# Import necessary libraries: OpenCV, time, and Mediapipe

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

# Import the Hand module from Mediapipe and necessary drawing utilities

cap = cv.VideoCapture(0)

# Open a video capture object (camera) with index 0 (default webcam)

# FPS part initialization
ctime = 0
ptime = 0

# Initialize variables for time tracking

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Read a frame from the camera, convert it to RGB, and process it with the Hand module

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    cv.circle(img, (cx, cy), 15, (255, 0, 0), cv.FILLED)
            mpdraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    # If hand landmarks are detected, iterate through them, extract information, and draw landmarks and connections

    # FPS
    ctime = time.time()
    if ptime != 0:
        fps = 1 / (ctime - ptime)
        cv.putText(img, f'FPS: {int(fps)}', (10, 45), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Calculate and display frames per second

    ptime = ctime
    cv.imshow('IMAGE', img)

    # Update time variables and display the image with landmarks

    key = cv.waitKey(1)
    
    if key == ord('q'):
        break

    # Wait for a key press, print the key value, and break the loop if 'q' is pressed

cap.release()
cv.destroyAllWindows()

# Release the video capture object and close all OpenCV windows when done
