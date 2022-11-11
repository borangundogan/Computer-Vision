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
    
    #Gaussin Blur for remove the noise
    img_blur = cv.GaussianBlur(frame,(3,3), sigmaX=1,sigmaY=1, borderType=cv.BORDER_DEFAULT)
    
    #Grayscale
    gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
    
    #CANNY EDGE DETECT
    canny_frame = cv.Canny(gray,70,120)    
    
    #Define parameters 
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    
    #SCHARR EDGE DETECT
    grad_x_scharr = cv.Scharr(gray,ddepth,dx = 1 ,dy = 0,scale=scale,delta=delta,borderType=cv.BORDER_DEFAULT)
    grad_y_scharr = cv.Scharr(gray,ddepth,dx = 0 ,dy = 1,scale=scale,delta=delta,borderType=cv.BORDER_DEFAULT)
    
    #Convert output to a CV_8U image (8 u-bit)
    abs_grad_x_scharr = cv.convertScaleAbs(grad_x_scharr)
    abs_grad_y_scharr = cv.convertScaleAbs(grad_y_scharr)
    
    #Gradient Approximate
    grad_scharr = cv.addWeighted(abs_grad_x_scharr, 0.5, abs_grad_y_scharr, 0.5, 0)
    
    #SOBEL EDGE DETECT
    #grad_sobel = cv.Sobel(gray,cv.CV_64F,dx = 1,dy = 1,ksize=5)
    
    grad_x_sobel = cv.Sobel(gray, ddepth, dx =2, dy =0, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y_sobel = cv.Sobel(gray, ddepth, dx= 0, dy= 2, ksize=5, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    #Convert output to a CV_8U image (8 u-bit)
    abs_grad_x_sobel = cv.convertScaleAbs(grad_x_sobel)
    abs_grad_y_sobel = cv.convertScaleAbs(grad_y_sobel)
    
    #Gradient Approximate
    grad_sobel = cv.addWeighted(abs_grad_x_sobel, 0.5, abs_grad_y_sobel, 0.5, 0)
    
    #LAPLACIAN EDGE DETECT
    laplacian = cv.Laplacian(gray,cv.CV_16S, ksize=5)
    abs_laplacian = cv.convertScaleAbs(laplacian)
    
    cv.imshow("Canny", canny_frame)
    cv.moveWindow("Canny", 0,0 )
    
    cv.imshow("Sobel", grad_sobel)
    cv.moveWindow("Sobel", 0,360)
    
    cv.imshow("Scharr", grad_scharr)
    cv.moveWindow("Scharr", 0,720)
    
    cv.imshow("Frame", gray)
    cv.moveWindow("Frame", 360,0)
    
    cv.imshow("Laplacian", abs_laplacian)
    cv.moveWindow("Laplacian", 360,360)
    
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
    