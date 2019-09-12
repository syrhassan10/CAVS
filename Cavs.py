import math

can_break = True

def break_distaance(velovity_eco_car, acceleration, distance_front):
    
    break_d = -(math.sqrt(velovity_eco_car))/(2*acceleration)
    if distance_front > break_d:
        can_break = True
    elif distance_front <= break_d:
        can_break = False

    return can_break




def main():
    print("your name")
main()

