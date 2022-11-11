import cv2 as cv
import sys

person = 1

height = 720
width = 480
fps = 120

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv.CAP_PROP_FPS,fps)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))

while True:
    _, frame = cap.read()

    if frame is None:
        sys.exit("Could not upload Frame!!")
    
    bounding_box_cordinates, weights =  cv.HOGCV.detectMultiScale(frame, winStride = (4, 4), padding = (8, 8), scale = 1.03)

    
    for x,y,w,h in bounding_box_cordinates:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.putText(frame, f'person {person}', (x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
        cv.putText(frame, 'Status : Detecting ', (40,40), cv.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv.putText(frame, f'Total Persons : {person-1}', (40,70), cv.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv.imshow('output', frame)
    
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    