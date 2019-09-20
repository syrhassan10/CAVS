import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

camera = cv2.VideoCapture ("project_video.mp4")
camera.open("project_video.mp4")
car_cascade = cv2.CascadeClassifier('cars.xml')
while True:
    (grabbed,frame) = camera.read()
    grayvideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(grayvideo, 1.1, 1)
    for (x,y,w,h) in cars:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
     cv2.imshow("video",frame)
    if cv2.waitKey(1)== ord('s'):
        break
camera.release()
cv2.destroyAllWindows()