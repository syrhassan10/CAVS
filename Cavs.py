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

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
cv2.imshow('result', gray_scale)
cv2.waitKey(0)

def break_distance(velovity_eco_car, acceleration, distance_front):

    break_d = -(math.sqrt(velovity_eco_car))/(2*acceleration)
    if distance_front > break_d:
        can_break = True
    elif distance_front <= break_d:
        can_break = False
    return can_break
