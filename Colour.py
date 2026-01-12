import cv2 as cv
import numpy as np
import math
v_path = '/home/pradyun/CV_practice/Assignment-4/Mallet_videos/IMG_9105.MOV'
cap = cv.VideoCapture(v_path)
fps =cap.get(cv.CAP_PROP_FPS)
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
while cap.isOpened():
    ret,frame= cap.read()
    if not ret:
        break
    image = frame.copy()
    S = cv.cvtColor(image, cv.COLOR_BGR2HSV)[:,:,1]
    S = cv.resize(S,(int(size[0]/2),int(size[1]/2)))
    cv.imshow("S channel",S)
    k = image.copy()
    k = cv.resize(k,(int(size[0]/2),int(size[1]/2)))
    cv.imshow("original",k)
    subtract = cv.cvtColor(frame, cv.COLOR_BGR2RGB)[:,:,2]
    red_filter = cv.cvtColor(frame, cv.COLOR_BGR2RGB)[:,:,0]
    red_filter = cv. resize(red_filter,(int(size[0]/2),int(size[1]/2)))
    red_filter = cv.threshold(red_filter, 230,255,cv.THRESH_BINARY)[1]
    frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB)[:,:,0]
    subtract = cv.threshold(subtract, 210,255,cv.THRESH_BINARY_INV)[1]
    frame = cv.resize(frame,(int(size[0]/2),int(size[1]/2)))
    frame = cv.normalize(frame, None, 0,255,cv.NORM_MINMAX)
    subtract = cv.resize(subtract,(int(size[0]/2),int(size[1]/2)))
    red_filter = cv.GaussianBlur(red_filter,(5,5),0)
    frame = cv.bitwise_and(frame,red_filter)
    frame = cv.bitwise_and(frame,subtract)
    kernel =cv.getGaussianKernel(5,5)
    kernel = np.outer(kernel,kernel)
    frame =cv.filter2D(frame, -1, kernel)
    #claheobj = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #frame = claheobj.apply(frame)
    new_img = cv.threshold(frame,110,255,cv.THRESH_BINARY)[1]
    frame = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    frame = cv.erode(frame, np.ones((3,3),np.uint8), iterations=1)
    image = cv.resize(image,(int(size[0]/2),int(size[1]/2)))
    #image = cv.Canny(image,100,200)
    counters, hierarchy = cv.findContours(frame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    min_area = 500
    max_area = 300009
    mask = np.zeros_like(frame)
    old_bounding_box = None
    old_area = 0
    i =0
    for count in counters :
        i += 1
        area = cv.contourArea(count) 
        bounding_rect = cv.boundingRect(count)
        if area > min_area and area < max_area:
            if old_area == 0:
                old_area = area
            if (abs(area - old_area) > 8000):
                continue
            if old_bounding_box is None:
                old_bounding_box = bounding_rect
            #if (abs(bounding_rect[0] - old_bounding_box[0]) > 300 or abs(bounding_rect[1] - old_bounding_box[1]) > 300):
            #   continue
            cv.drawContours(mask, [count], -1, 255, thickness= cv.FILLED)
            old_bounding_box = bounding_rect
            old_area = area
    mask = cv.bitwise_and(mask,new_img)
    cv.imshow("thresholded",new_img)
    image = cv.bitwise_and(image,image,mask=mask)
    cv.imshow("cropped",image)
    cv.waitKey(10)
cap.release()
cv.destroyAllWindows()
print(f"The cropped video is saved as {cap} at size {size} and fps {fps}")