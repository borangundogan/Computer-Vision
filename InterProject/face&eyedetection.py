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

cascPath = "haar/haarcascade_frontalface_default.xml"
eyePath = "haar/haarcascade_eye.xml"

faceCascade = cv.CascadeClassifier(cascPath)
eyeCascade = cv.CascadeClassifier(eyePath)


while cap.isOpened():
    _, frame = cap.read()
    
    if frame is None:
        sys.exit("Could not upload Frame!!")
    
    #Grayscale
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #scaleFactor : Parameter specifying how much the image size is reduced at each image scale. (default = 1.1)
    #minNeighbors : Parameter specifying how many neighbors each candidate rectangle should have to retain it. (default = 5)
    
    faces = faceCascade.detectMultiScale(gray_img, 1.3, 4, cv.CASCADE_SCALE_IMAGE)
    eyes = eyeCascade.detectMultiScale(gray_img)
    
 
    for face in faces:
        (x, y, w, h) = face
        
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        
        cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 3)
        cv.putText(frame,"Unsigned Person", (x - 5, y - 5), cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
        
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),3)
    
    cv.imshow("Frame", frame)
    cv.moveWindow("Frame", 0,0)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()