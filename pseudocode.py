import math
import numpy as np
import matplotlib.pyplot as plt


def break_distance(velovity_eco_car, acceleration, pd_straight_distance, current_wheel_rotations):
    
    break_d = -(math.sqrt(velovity_eco_car))/(2*acceleration)
    if pd_straight_distance > break_d:
        can_break = True
        pd_straight_distance(1,1,1, break_d,pd_straight_distance,current_wheel_rotations) #call the following function. kp kd ki can be tuned with further testing
    elif pd_straight_distance <= break_d:
        can_break = False
    return can_break

#checks for surrounding open lanes/ plains in case of unavoidable collision
def collision(can_break, front_surround, right_surround, left_surround, back_surround):
    if (can_break == False) and front_surround == True : #if front is fully blocked off
        if right_surround == True and left_surround == True: #if their are cars or no empty space beside car 
            warning = True
            air_bags = True
            brakes = True
            hand_break = True
            call_help = True
            collision = True
            return collision
        #in case of empty space in front and behind the car will attempt to drift
        if right_surround == False and left_surround == False and back_surround == False: #drifts
            turn_right = True #turns right
            hand_break = True
            breaks = True
            call_help = True
            drift = True
            return drift
            
    if (can_break == False) and front_surround == False : #if there is space beside the car it is about to collide into
        if right_surround == False:
            turn_right_lane = True #truns to right lane
            return turn_right_lane
        if left_surround == True:
            turn_left_lane = True #turns to right lane
            return turn_left_lane

def blindness(sensor_input, pre_back_surround):
    if sensor_input == False: #no sensor input
        emergency_lights = True
        warning = True
        if pre_back_surround == False:
            breaks = True


def app_relative(lst):
    #calculates it based on realtivity if the surface is increasing compared to our speed and the approaching car's speed
    approach = True

def incoming_car(surface):
    if surface[0] > surface[1]:
        object_app = True
        if app_relative(surface) == True:
            car_app = True
    return car_app








