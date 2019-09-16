import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

#global
can_break = True

#lane detecting module
def canny(image):
    gray_scale = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussainBlur(gray_scale, (5,5), 0)
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
    polygons = np.array([(200, height), (1100, height), (550, 250)])
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
combo_image = cv2.addWeighted(lane_image, 0.8)
cv2.imshow("result", cropped_image)
cv2.waitKey(0)
plt.imshow(canny)
plt.show(0)


def break_distance(velovity_eco_car, acceleration, distance_front, current_wheel_rotations):

    break_d = -(math.sqrt(velovity_eco_car))/(2*acceleration)
    if distance_front > break_d:
        can_break = True
        pd_straight_distance(1,1,1, break_d,distance_front,current_wheel_rotations) #call the following function. kp kd ki can be tuned with further testing
    elif distance_front <= break_d:
        can_break = False
    return can_break

def pd_straight_distance(kp, ki, kd, break_distance, current_distance_front,current_wheel_rotations):
    previous_error = 0
    wheelDiameter = 1 #assume wheel diamter is 1 meter
    integral = 0
    required_wheel_rotations = break_distance/ (wheelDiameter * 3.14)
    reset_currentWheelRotation = True #resests our current reading of wheel rotation to 0
    while(current_wheel_rotations <= required_wheel_rotations): #keeps running until distance front = breaking distance
         #calculates the amount of distance needed to reach breaking distance
        error = break_distance - current_distance_front #set point = breaking distance (desired front distance)
        propotional = error *kp 
        # since the error is propotional to the correction, the greater the error the greater the correction (thus faster) 
        #thus we can introduce a constant which is used to scale up or down our error. (if kp is too high, vehicle may overshoot)
        deltaError = error - previous_error
        derivative  = deltaError*kd 
        #the job of the derivative is to slow down the rate of change in error over time. 
        # this is used to smooth our correction i.e. overshoot less(since derivative is the rate of change)
        if ((error >= 0) & (break_distance > current_distance_front)): #if true you can speed up. gas throttle will be controlled using a seperate servo motor
            gas_servo_motor = derivative + propotional
        else: #otherwise we need to slow down. break throttle will be controlled using a seperate servo motor
            breaking_servo_motor = derivative + propotional
        



# the bottom 2 lines will work if only we have a video where there are cars
camera = cv2.VideoCapture ("video.avi")
camera.open("video.avi")
car_cascade = cv2.CascadeClassifier('cars.xml')
while True:
    (grabbed,frame) = camera.read()
    grayvideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(grayvideo, 1.1, 1)
    for (x,y,w,h) in cars:
     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
     cv2.imshow("video",frame)
    if cv2.waitKey(1)== ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
