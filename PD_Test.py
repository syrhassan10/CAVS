def pd_straight_distance(break_distance, current_distance_front):
    kp = 0
    kd = 0
    previous_error = 0

    break_distance = 20
    current_distance_front = 5
    velocity_car = 90
    error = break_distance - current_distance_front
    while error != 0:

        error = break_distance - current_distance_front  # set point = breaking distance (desired front distance)
        proportional = error * kp
        deltaError = error - previous_error
        derivative = deltaError * kd
        if (error >= 0) & (break_distance > current_distance_front):
            break_pedal = True
            gas_throttle = False
            breaking_servo_motor = derivative + proportional
            current_distance_front = current_distance_front + 1
            print("Error :", error)

        else:
            gas_throttle = True
            break_pedal = False
            gas_servo_motor = derivative + proportional

