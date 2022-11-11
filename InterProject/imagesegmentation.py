#Created by Boran Gundogan --- https://www.linkedin.com/in/boran-gundogan/
import cv2 as cv
import sys
import numpy as np


height = 720
width = 480
fps = 120

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv.CAP_PROP_FPS,fps)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))


while cap.isOpened():
    _, frame = cap.read()
    
    if frame is None:
        sys.exit("Could not upload Frame!!")
    
    hsv_img = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    #I look threshold func., because, ı wonder how it is work with image segmentation,but it works very well with edge detection.
    _,thresh = cv.threshold(hsv_img, 60,255, cv.THRESH_BINARY_INV)
    
    #Define the Ranges for Orange, set your own values
    low_range = np.array([12,219,0])
    high_range = np.array([61,255,255])
        
    my_mask = cv.inRange(hsv_img,low_range,high_range)
    
    #Find the contours    
    contours, junk = cv.findContours(my_mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv.contourArea(contour)
        
        if area >= 100:
            cv.drawContours(frame,[contour], -1,(0,0,255), 2)
            
    cv.imshow("Frame", frame)
    cv.moveWindow("Frame", 0,0)
    
    cv.imshow("thresh", thresh)
    cv.moveWindow("thresh", 0,480)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()