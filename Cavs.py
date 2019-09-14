import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

#global
can_break = True
cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#lane detecting module
def canny(image):
    gray_scale = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_scale, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
            print(line)
    return line_image



def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength= 40, maxLineGap=5)
line_image = display_lines(lane_image, lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow("result", combo_image)
cv2.waitKey(0)
plt.imshow(canny)
plt.show(0)


def break_distance(velovity_eco_car, acceleration, distance_front):

    break_d = -(math.sqrt(velovity_eco_car))/(2*acceleration)
    if distance_front > break_d:
        can_break = True
    elif distance_front <= break_d:
        can_break = False
    return can_break



# capture frames from a video(if we dont have any video we can remove this line)
cap = cv2.VideoCapture('video.avi') 
  
# Trained XML classifiers describes some features of some object we want to detect 
car_cascade = cv2.CascadeClassifier('cars.xml') 
  
# loop runs if capturing has been initialized. 
while True: 
    # reads frames from a video 
    ret, frames = cap.read() 
      
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
      
  
    # Detects cars of different sizes in the input image 
    cars = car_cascade.detectMultiScale(gray, 1.1, 1) 
      
    # To draw a rectangle in each cars 
    for (x,y,w,h) in cars: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2) 
  
   # Display frames in a window  
   cv2.imshow('video2', frames) 
      
    # Wait for Esc key to stop(if we are not using a video this line can also be removed) 
    if cv2.waitKey(33) == 27: 
        break