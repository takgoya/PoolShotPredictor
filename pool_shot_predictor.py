import cv2
import numpy as np
import math

from utils import calculate_distance, get_line_points, check_line_intersection, reflect_line, extend_line

def find_table(image):
    '''
    Finds the table based on HSV values.
    These values are obtained from ColorPicker
    '''
    # HSV values obtained from ColorPicker
    hue_min, hue_max = 58, 80
    sat_min, sat_max = 125, 250
    val_min, val_max = 92, 255

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    hsv_mask = cv2.inRange(hsv_image, lower, upper)
    
    return hsv_mask


def find_table_holes(image):
    '''
    Finds the table holes/pockets.
    Using contours, it also makes an erosion to help on detecting the holes.
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    table_erode = cv2.erode(image, kernel, iterations=2)

    contours, _ = cv2.findContours(table_erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    x, y, w, h = cv2.boundingRect(contours[0])

    top_left = (x, y)
    top_cent = (x+(w//2), y)
    top_right = (x+w, y)
    bot_left = (x, y+h)
    bot_cent = (x+(w//2),y+h)
    bot_right = (x+w, y+h)

    return [top_left, top_cent, top_right, bot_left, bot_cent, bot_right]


def find_cue_ball_stick(image):
    '''
    Finds the cue ball and the stick based on contours.
    It filters on contourArea and approx.
    Depending on the position of the stick it returns the best approach for the stick.
    It means its top or center position.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 1)
    _, thresh = cv2.threshold(blur, 127, 150, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    cue_ball_center = None
    cue_ball_radius = None
    cue_ball_contour = None
    stick_contour = None
    stick = None
    direction_up = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02*cv2.arcLength(contour, True), True)
        if (len(approx) == 8) and 700 < cv2.contourArea(contour) < 800:
            cue_ball_contour = contour
        elif len(approx) < 8 and 300 < cv2.contourArea(contour) < 3000:
            stick_contour = contour

    if cue_ball_contour is not None and stick_contour is not None:
        cue_ball_center, cue_ball_radius = cv2.minEnclosingCircle(cue_ball_contour)
        
        rect = cv2.minAreaRect(stick_contour)
        box_stick = cv2.boxPoints(rect)
        box_stick = np.intp(box_stick)
        top_left, top_right, bot_right, bot_left = box_stick

        if calculate_distance(top_left, top_right) < calculate_distance(top_left, bot_left):
            stick_top_cent = (top_left[0]+top_right[0])/2, (top_left[1]+top_right[1])/2
            stick_bot_cent = (bot_left[0]+bot_right[0])/2, (bot_left[1]+bot_right[1])/2
            if stick_top_cent[1] > cue_ball_center[1]:
                direction_up = True
                stick = stick_top_cent
            else:
                direction_up = False
                stick_center = stick_top_cent[0] + (stick_bot_cent[0]-stick_top_cent[0])/2, stick_top_cent[1] + (stick_bot_cent[1]-stick_top_cent[1])/2
                stick = stick_center
        else:
            stick_top_cent = (top_right[0]+bot_right[0])/2, (top_right[1]+bot_right[1])/2
            stick_bot_cent = (top_left[0]+bot_left[0])/2, (top_left[1]+bot_left[1])/2
            if stick_top_cent[1] > cue_ball_center[1]:
                direction_up = True
            else:
                direction_up = False
            stick_center = stick_top_cent[0] + (stick_bot_cent[0]-stick_top_cent[0])/2, stick_top_cent[1] + (stick_bot_cent[1]-stick_top_cent[1])/2
            stick = stick_bot_cent

    return cue_ball_center, cue_ball_radius, stick, direction_up


def detect_target_balls(image, excluded):
    '''
    Detects the target balls baed on HoughCircles.
    It receives a list of excluded circles that are the pockets and the cue ball.
    Those are discarded and the correct target ball is returned.
    '''
    ball = None
    ball_radius = None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # detect circles in the image
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, minDist=20,
                               param1=100, param2=20, minRadius=15, maxRadius=15)
    
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            is_excluded = False
            for exc in excluded:
                distance = calculate_distance(exc, (x,y))
                if distance < 10:
                    is_excluded = True
            if is_excluded == False:
                ball = (x,y)
                ball_radius = r
    return ball, ball_radius


def get_collision_point(cue_ball_position,
                        cue_ball_radius,
                        target_ball_position,
                        target_ball_radius,
                        stick_position,
                        table_limits,
                        direction_up):
    '''
    Finds the collision point between the cue ball and the target ball.
    It projects a line from the stick through the cue_ball center position.
    Along that line it calculates all the points to simulate the cue_ball movement.
    It checks for the intersection with the target ball based on the distance between the centers.
    '''

    collision_point = None

    stick_cue_min, stick_cue_max = extend_line(stick_position, cue_ball_position, table_limits)

    if direction_up:
        trajectory_points = get_line_points(stick_position, stick_cue_min)
        # trajectory_points = bresenham_line(stick_position, stick_cue_min)
    else:
        trajectory_points = get_line_points(stick_position, stick_cue_max)
        # trajectory_points = bresenham_line(stick_position, stick_cue_max)

    for idx, pt in enumerate(trajectory_points):
        r = cue_ball_radius + target_ball_radius
        # dist = np.sqrt((target_ball_position[0] - pt[0])**2 + (target_ball_position[1] - pt[1])**2)
        dist = calculate_distance(target_ball_position, pt)
        if dist <= r:
            if idx != 0:
                collision_point = trajectory_points[idx-1]
            else:
                collision_point = trajectory_points[idx]
            break

    return collision_point


def predict_final_position(target_ball_position,
                           target_ball_radius,
                           collision_point,
                           table_sides,
                           table_limits,
                           direction_up):
    '''
    Predicts the final position of the ball.
    It calculates the projected line of the target ball from the collision point.
    That line is evaluated against the table sides to see if it intersects, to obtain that final position.
    '''
    predicted_min, predicted_max = extend_line(collision_point, target_ball_position, table_limits)
    
    if direction_up:
        intersection_point, _ = check_line_intersection(target_ball_position, predicted_min, table_sides, table_limits)
    else:
        intersection_point, _ = check_line_intersection(target_ball_position, predicted_max, table_sides, table_limits)
    
    return intersection_point



def check_ball_in(predicted_point, table_holes, hole_radius=24):
    '''
    Checks if the ball is in or not.
    It is done trhough comparing the distance of the predicted final position of the target ball against all the pocket holes.
    Based on the distance of the pocket hole and the target ball, it returns if the ball is in or not.
    '''
    for idx, hole_pos in enumerate(table_holes):
        dist = calculate_distance(predicted_point, hole_pos)

        if dist <= hole_radius:
            return True
    
    return False


def make_prediction(target_ball_position, collision_point, table_sides_up, table_sides_down, table_limits, table_holes, direction_up):
    '''This function makes the prediction of the target ball.
    If a collision point exits between the cue ball and the target ball, it calculates the final point of the target ball.
    It checks if the final position is inside of any of the holes.
    In case that there is not inside of any of the pocket holes, it tries to calculate the reflection of the projected line of the target ball.
    It checks against the different table sides to find a possible intersection.
    If it does, it then checks the new intersection point originated by the reflected  line, and that new predicted end point is evaluated.
    If it still is not inside a pocket hole, it tries a final possible reflection (2 reflection max)
    '''
    predicted_position = None
    reflected_predicted_position = None
    is_ball_in = False

    if direction_up:
        predicted_position = predict_final_position(target_ball_position,
                                            target_ball_radius,
                                            collision_point,
                                            table_sides_up,
                                            table_limits,
                                            direction_up)
    else:
        predicted_position = predict_final_position(target_ball_position,
                                            target_ball_radius,
                                            collision_point,
                                            table_sides_down,
                                            table_limits,
                                            direction_up)
    
    if predicted_position is not None:
        is_ball_in = check_ball_in(predicted_position, table_holes)
        if not is_ball_in:
            # predicted_line = np.array([[predicted_position[0], predicted_position[1]],[target_ball_position[0],target_ball_position[1]]])
            if direction_up:
                intersection_point, intersection_side = check_line_intersection(target_ball_position, predicted_position, table_sides_up, table_limits)
                reflection_line = reflect_line(target_ball_position, intersection_point, intersection_side[0], intersection_side[1])
                reflected_predicted_position, _ = check_line_intersection(reflection_line[0], reflection_line[1], table_sides_down, table_limits)
                if intersection_point is not None and reflected_predicted_position is not None:
                    is_ball_in = check_ball_in(reflected_predicted_position, table_holes) 
            else:
                intersection_point, intersection_side = check_line_intersection(target_ball_position, predicted_position, table_sides_down, table_limits)
                reflection_line = reflect_line(target_ball_position, intersection_point, intersection_side[0], intersection_side[1])
                reflected_predicted_position, _ = check_line_intersection(reflection_line[0], reflection_line[1], table_sides_up, table_limits)
                if intersection_point is not None and reflected_predicted_position is not None:
                    is_ball_in = check_ball_in(reflected_predicted_position, table_holes)
                    if not is_ball_in:
                        if intersection_point is not None:
                            reflect_intersection_point, reflect_intersection_side = check_line_intersection(intersection_point, reflected_predicted_position, table_sides_down, table_limits)
                            reflection_line = reflect_line(target_ball_position, reflect_intersection_point, reflect_intersection_side[0], reflect_intersection_side[1])
                            reflected_predicted_position, reflect_predicted_side = check_line_intersection(reflection_line[0], reflection_line[1], table_sides_up, table_limits)
                            if intersection_point is not None and reflected_predicted_position is not None:
                                is_ball_in = check_ball_in(reflected_predicted_position, table_holes)
                                if is_ball_in:
                                    if reflect_intersection_side[1][0] == reflect_intersection_side[1][0] and reflect_intersection_side[1][1] == reflect_intersection_side[1][1]:
                                        predicted_position = reflected_predicted_position
    
    return predicted_position, reflected_predicted_position, is_ball_in


def draw_prediction(cue_ball_position,
                    cue_ball_radius,
                    collision_point,
                    predicted_position,
                    reflected_predicted_position,
                    is_ball_in,
                    table_limits):
    '''
    Draws the prediction of the shot.
    Based on the result, IN or OUT, it draws on different colors.
    It also adds the text for the prediciton.
    '''
    if is_ball_in:
        color = (0,255,0)
        text = 'Prediction: IN'
    else:
        color = (0,0,255)
        text = 'Prediction: OUT'

    xmin, ymin, xmax, ymax = table_limits
    
    cv2.line(frame, (int(cue_ball_position[0]), int(cue_ball_position[1])), (int(collision_point[0]), int(collision_point[1])), color, 2)
    cv2.circle(frame, (int(collision_point[0]), int(collision_point[1])), int(cue_ball_radius), color, cv2.FILLED)
    cv2.line(frame, (int(collision_point[0]), int(collision_point[1])), (int(predicted_position[0]), int(predicted_position[1])), color, 2)
    cv2.circle(frame, (int(predicted_position[0]), int(predicted_position[1])), int(cue_ball_radius), color, cv2.FILLED)
    if reflected_predicted_position:
        cv2.line(frame, (int(predicted_position[0]), int(predicted_position[1])), (int(reflected_predicted_position[0]), int(reflected_predicted_position[1])), color, 2)
        cv2.circle(frame, (int(reflected_predicted_position[0]), int(reflected_predicted_position[1])), int(cue_ball_radius), color, cv2.FILLED)

    if collision_point[0] < (xmax - xmin)/2:
        cv2.rectangle(frame, (collision_point[0] + 50, collision_point[1] - 15), (collision_point[0] + 250, collision_point[1] + 10), color, -1)
        cv2.putText(frame, text, (collision_point[0] + 95, collision_point[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
    else:
        cv2.rectangle(frame, (collision_point[0] - 250, collision_point[1] - 15), (collision_point[0] - 50, collision_point[1] + 10), color, -1)
        cv2.putText(frame, text, (collision_point[0] - 210, collision_point[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)



video_path = 'video/Shot-Predictor-Video.mp4'

cap = cv2.VideoCapture(video_path)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

resized_w, resized_h = 1280, 720

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('prediction.avi', fourcc, fps, (resized_w, resized_h))

table_xmin, table_ymin, table_xmax, table_ymax = 0, 0, 0, 0
is_moving = True

# vars to fix a problem updating the last table, to avoid clearing the prediction
# for display purposes, does not affect the result
table_updates = 0
max_table_updates = 16

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (resized_w,resized_h))
  
    table = find_table(frame)

    table_holes = find_table_holes(table)
    table_top_left, table_top_cent, table_top_right, table_bot_left, table_bot_cent, table_bot_right = table_holes

    table_top = np.array([[table_top_left[0],table_top_left[1]],[table_top_right[0], table_top_right[1]]])
    table_bot = np.array([[table_bot_left[0],table_bot_left[1]],[table_bot_right[0], table_bot_right[1]]])
    table_left = np.array([[table_top_left[0],table_top_left[1]],[table_bot_left[0], table_bot_left[1]]])
    table_right = np.array([[table_top_right[0],table_top_right[1]],[table_bot_right[0], table_bot_right[1]]])
    table_sides = [table_left, table_top, table_right, table_bot]
    table_sides_up = [table_left, table_top, table_right]
    table_sides_down = [table_left, table_bot, table_right]
    
    # update values, new table
    if table_xmin!=table_top_left[0] and table_ymin!=table_top_left[1] and table_xmax!=table_bot_right[0] and table_ymax!=table_bot_right[1] and table_updates != max_table_updates:
        table_updates += 1

        table_xmin, table_ymin = table_top_left[0], table_top_left[1]
        table_xmax, table_ymax = table_bot_right[0], table_bot_right[1]

        cue_ball_position = None
        cue_ball_radius = None
        target_ball_position = None
        target_ball_radius = None
        stick_position = None
        collision_point = None
        predicted_position = None
        reflected_predicted_position = None
        is_moving = False
        is_ball_in = False
        direction_up = None
        prediction_done = False

    table_limits = [table_xmin, table_ymin, table_xmax, table_ymax]

    # if cue_ball is None or target_ball is None or stick_top is None or stick_bot is None:
    table_mask = np.zeros_like(frame)
    cv2.rectangle(table_mask, table_top_left, table_bot_right, (255,255,255), -1)

    # define ROI
    roi_frame = cv2.bitwise_and(frame, table_mask)
    
    # if cue_ball is not moving, recalculate params
    if not is_moving:
        if prediction_done and direction_up is not True:
            draw_prediction(cue_ball_position, cue_ball_radius, collision_point, predicted_position, reflected_predicted_position, is_ball_in, table_limits)
        else:
            ball, ball_radius, stick, direction = find_cue_ball_stick(roi_frame)

            # set cue_ball
            if cue_ball_position is None and ball is not None:
                cue_ball_position = ball
                cue_ball_radius = int(ball_radius)

            # set stick and direction; only update if there is a new valid value
            if stick is not None and direction is not None:
                if stick_position is None or direction is True:
                    stick_position = stick
                    direction_up = direction                

            # set target_ball
            if ball is not None and cue_ball_position is not None:
                if target_ball_position is None:
                    excluded = table_holes
                    excluded.append(cue_ball_position)
                    target_ball_position, target_ball_radius = detect_target_balls(roi_frame, excluded)

                if cue_ball_position is not None and target_ball_position is not None and stick_position is not None:

                    if not prediction_done or (prediction_done and direction_up):
                        collision = get_collision_point(cue_ball_position, cue_ball_radius, target_ball_position, target_ball_radius, 
                                                        stick_position, table_limits, direction_up)
                        
                        if collision is not None:
                            collision_point = collision

                            predicted_position, reflected_predicted_position, is_ball_in = make_prediction(target_ball_position, collision_point, table_sides_up, 
                                                                                                     table_sides_down, table_limits, table_holes, direction_up)

                            if predicted_position is not None:
                                prediction_done = True

                move_x = abs(ball[0] - cue_ball_position[0])
                move_y = abs(ball[1] - cue_ball_position[1])

                if move_x > 1 or move_y > 1:
                    is_moving = True

                    draw_prediction(cue_ball_position, cue_ball_radius, collision_point, predicted_position, reflected_predicted_position, is_ball_in, table_limits)

    if is_moving:
        draw_prediction(cue_ball_position, cue_ball_radius, collision_point, predicted_position, reflected_predicted_position, is_ball_in, table_limits)   

    cv2.imshow('Pool', frame)

    output.write(frame)

    if cv2.waitKey(fps) & 0xFF==ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()