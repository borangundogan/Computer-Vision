#Created by Boran Gundogan --- https://www.linkedin.com/in/boran-gundogan/
import cv2 as cv
import sys



height = 720
width = 480
fps = 120

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv.CAP_PROP_FPS,fps)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

cascPath = "haar/haarcascade_frontalface_default.xml"

faceCascade = cv.CascadeClassifier(cascPath)


while cap.isOpened():
    _, frame = cap.read()
    
    if frame is None:
        sys.exit("Could not upload Frame!!")
    
    #Grayscale
    gray_img = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
   
    faces = faceCascade.detectMultiScale(gray_img, 1.3, 4, cv.CASCADE_SCALE_IMAGE)

    for (x, y, w, h)  in faces:
        
        #cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 3)
        
        #Median Blur Method
        frame[y:y+h, x:x+w]  = cv.medianBlur(frame[y:y+h, x:x+w],35)
        
        #Gaussian Blur Method
        face = frame[x:x+w, y:y+h]
        face_gaussian = cv.GaussianBlur(face,(23, 23),20)
        
        cv.putText(frame,"Unsigned Person", (x - 5, y - 5), cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            
    cv.imshow("Frame", frame)
    cv.moveWindow("Frame", 0,0)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()