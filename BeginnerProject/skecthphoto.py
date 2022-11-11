#Created by Boran Gundogan --- https://www.linkedin.com/in/boran-gundogan/
import cv2 as cv
import sys

height = 720
width = 360
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
    
    #Grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Invert Image
    invert_img = cv.bitwise_not(gray)
    #invert = 255 - gray
    
    #Gaussin Blur for remove the noise
    img_blur = cv.GaussianBlur(invert_img,(21,21), 0)
    
    #Sketch the Image
    skecth_img = cv.divide(gray, 255 - img_blur,scale=255.0)
    
    cv.imshow("Frame", frame)
    cv.moveWindow("Frame", 0,0)
    
    cv.imshow("Skecth", skecth_img)
    cv.moveWindow("Skecth", 0,360)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    