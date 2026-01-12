import cv2 as cv
import numpy as np
import math
import cv2.aruco as aruco
v_path = '/home/pradyun/IP_Assignment/Assignment-4/ArUco_videos/Video10(4x4_250).mp4'
cap = cv.VideoCapture(v_path)
fps =cap.get(cv.CAP_PROP_FPS)
size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
while cap.isOpened():
    ret,frame= cap.read()
    if not ret:
        break
    kernel =np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    frame =cv.filter2D(frame, -1, kernel)
    frame = cv.filter2D(frame, -1, kernel)
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    arucoParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(frame)
    if len(corners) > 0:
        aruco.drawDetectedMarkers(frame, corners, ids)
    cv.imshow("Aruco Detection", frame)
    cv.waitKey(10)
cap.release()
cv.destroyAllWindows()
print(f"The processed video is saved as {cap} at size {size} and fps {fps}")