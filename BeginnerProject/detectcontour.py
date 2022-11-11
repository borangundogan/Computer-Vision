#Created by Boran Gundogan --- https://www.linkedin.com/in/boran-gundogan/
import cv2 as cv
import sys
import numpy as np

#Callbacks
def onTrack1(val) -> None:
    global hueLow
    hueLow = val
    print("Hue Low", hueLow)

def onTrack2(val) -> None:
    global hueHigh
    hueHigh = val
    print("Hue High", hueHigh)
    
def onTrack3(val) -> None:
    global satLow
    satLow = val
    print("Sat Low", satLow)
    
def onTrack4(val) -> None:
    global satHigh
    satHigh = val
    print("Sat High", satHigh)

def onTrack5(val) -> None:
    global valLow
    valLow = val
    print("val Low", valLow)
    
def onTrack6(val) -> None:
    global valHigh
    valHigh = val
    print("Val High", valHigh)

height = 720
width = 360
fps = 120

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv.CAP_PROP_FPS,fps)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

cv.namedWindow("myTracker")
cv.moveWindow("myTracker", width + 10,0)

#default parameters
hueLow = 10
hueHigh = 20
satLow = 10
satHigh = 250
valLow = 10
valHigh = 250

cv.createTrackbar("Hue Low" , "myTracker", 10, 179 ,onTrack1)
cv.createTrackbar("Hue High" , "myTracker", 20, 179 ,onTrack2)
cv.createTrackbar("Sat Low" , "myTracker", 10, 255 ,onTrack3)
cv.createTrackbar("Sat High" , "myTracker", 250, 255 ,onTrack4)
cv.createTrackbar("Val Low" , "myTracker", 10, 255 ,onTrack5)
cv.createTrackbar("Val High" , "myTracker", 250, 255 ,onTrack6)



while cap.isOpened():
    _, frame = cap.read()
    
    if frame is None:
        sys.exit("Could not upload Frame!!")
        
    #Convert from BGR to HSV 
    hsv_img = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    #Define the Ranges for low and high
    low_range = np.array([hueLow,satLow,valLow])
    high_range = np.array([hueHigh,satHigh,valHigh])
        
    my_mask = cv.inRange(hsv_img,low_range,high_range)
    
    #Find the contours    
    contours, junk = cv.findContours(my_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    #Draw the contours
    cv.drawContours(frame,contours,-1,(255,0,0))
    
    cv.imshow("Frame", frame)
    cv.moveWindow("Frame", 0,0)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()