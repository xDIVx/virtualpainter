# Import necessary libraries
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm  # Assuming there's a module named HandTrackingModule with hand detection functions

# Set brush and eraser thickness
brushThickness = 25
eraserThickness = 100

# Specify the folder path containing images
folderPath = "D:\Virtual_Painter-master\Virtual_Painter-master\image"

# Get a list of image files in the specified folder
myList = os.listdir(folderPath)
print(myList)

# Create a list to store image overlays
overlayList = []

# Load each image in the folder and append it to the overlay list
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# Print the number of images loaded
print(len(overlayList))

# Set the initial header image
header = overlayList[0]
drawColor = (255, 0, 255)  # Set initial drawing color

# Open the webcam using OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detection module
detector = htm.handDetector(detectionCon=0.65, maxHands=1)

# Initialize variables for previous hand position
xp, yp = 0, 0

# Create an empty canvas for drawing
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Main loop for real-time drawing
while True:
    # Read a frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the frame horizontally for a mirror effect

    # Detect hands in the frame
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Check if any hand landmarks are detected
    if len(lmList) != 0:
        print(lmList)
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Detect the state of fingers using hand landmarks
        fingers = detector.fingersUp()

        # Selection mode based on finger positions
        if fingers[1] and fingers[2]:
            print("Selection Mode")

        # Change header and drawing color based on finger positions
        if y1 < 125:
            if 250 < x1 < 450:
                header = overlayList[0]
                drawColor = (255, 0, 255)
            elif 550 < x1 < 750:
                header = overlayList[1]
                drawColor = (255, 0, 0)
            elif 800 < x1 < 950:
                header = overlayList[2]
                drawColor = (0, 255, 0)
            elif 1050 < x1 < 1200:
                header = overlayList[3]
                drawColor = (0, 0, 0)

        # Draw a rectangle indicating the selected color
        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # Drawing mode based on finger positions
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

        # Draw a line on the canvas
        if xp == 0 and yp == 0:
            xp, yp = x1, y1

        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)

        xp, yp = x1, y1

        # Prepare the canvas for overlaying with the webcam frame
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Display the header at the top of the frame
        img[0:125, 0:1280] = header

        # Display the resulting frames
        cv2.imshow("Image", img)
        cv2.imshow("Canvas", imgCanvas)
        cv2.imshow("Inv", imgInv)

        # Wait for a key press (1 millisecond delay)
        cv2.waitKey(1)
