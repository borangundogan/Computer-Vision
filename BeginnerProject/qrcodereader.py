import cv2 as cv
import numpy as np
import sys
from pyzbar.pyzbar import decode


def main(frame) -> None:
    #GrayScale
    gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #Pyzbar decode function
    code = decode(gray_frame)
    
    for obj in code:
        # Get points and rect for points
        points = obj.polygon
        (x,y,w,h) = obj.rect
        
        #pts: Array of polygonal curves. , isClosed: Flag indicating whether the drawn polylines are closed or not
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1,1,2))
        
        print(pts)
        
        #Cv polylines
        cv.polylines(frame,[pts], True,(0,0,255), 3)
        
        barcodeData = obj.data.decode("utf-8")
        barcodeType = obj.type
        
        string = "Data " + str(barcodeData) + " | Type " + str(barcodeType)
        
        cv.putText(frame, string, (x,y), cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)
        
        print("Barcode: "+barcodeData +" , Type: "+barcodeType)
    return None

if __name__ == "__main__":
    height = 720
    width =  480
    fps = 60
    
    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,height)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv.CAP_PROP_FPS,fps)
    
    while cap.isOpened():
        
        _, frame = cap.read()
        
        if frame is None:
            sys.exit("Could not reach the frame!")
        
        main(frame)
        cv.imshow("Frame", frame)
        
        
        if cv.waitKey(1) & 0xff == 'q':
            break
    
    cap.release()
    cv.destroyAllWindows()