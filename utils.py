import math
import cv2
import numpy as np


def calculate_distance(point1, point2):
    '''Calculate the distance between two points.'''
    x1, y1 = point1
    x2, y2 = point2

    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance


def extend_line(point1, point2, limits):
    '''Extends a line that goes from point1 to point2 towards some given limits.'''
    x1, y1 = point1
    x2, y2 = point2
    xmin, ymin, xmax, ymax = limits

    if x1 == x2:  # vertical line
        x_min = x1
        x_max = x1
        y_min = max(ymin, min(y1, y2))
        y_max = min(ymax, max(y1, y2))
    elif y1 == y2:  # horizontal line
        x_min = max(xmin, min(x1, x2))
        x_max = min(xmax, max(x1, x2))
        y_min = y1
        y_max = y1
    else:  # non-vertical, non-horizontal line      
        # calculate the slope and y-intercept of the line
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        x_min = max(xmin, (ymin - b) / m)
        x_max = min(xmax, (ymax - b) / m)
        y_min = max(ymin, m * xmin + b)
        y_max = min(ymax, m * xmax + b)
        
        if x1 > x2 and y1 < y2:            
            x_min = (ymin - b) / m
            x_max = (ymax - b) / m
            y_min = ymin
            y_max = ymax
    
    # return the extended line as two points
    return (int(x_min), int(y_min)), (int(x_max), int(y_max))


def get_line_points(point1, point2):
    '''Get the line points that are between point1 and point2'''
    # Get the coordinates of the two points
    x1, y1 = point1
    x2, y2 = point2

    # Calculate the slope and y-intercept of the lines
    if (x2 - x1) == 0:
        slope = 0
    else:
        slope = (y2 - y1) / (x2 - x1)
    y_intercept = y1 - slope * x1

    # Determine the number of points needed and the x increment
    num_points = max(abs(x2 - x1), abs(y2 - y1))
    x_increment = (x2 - x1) / num_points

    # Generate the list of points on the line
    line_points = []
    for i in range(int(num_points) + 1):
        x = int(x1 + i * x_increment)
        y = int(slope * x + y_intercept)
        line_points.append((x, y))
        # print(x, y)

    line_points = list(dict.fromkeys(line_points))    

    return line_points


def get_slope_intercept(point1, point2):
    '''Gets the slope and intercept of a line'''
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2: # Vertical Line
        slope = None
        intercept = x1
    elif y1 == y2: # Horizontal Line
        slope = 0
        intercept = y1
    else: # Regular Line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    return slope, intercept


def get_line_intersection(point1, point2, point3, point4):
    '''Gets the intersection point of two lines'''
    slope1, intercept1 = get_slope_intercept(point1, point2)
    slope2, intercept2 = get_slope_intercept(point3, point4)
    
    if slope1 == slope2:
        # The lines are parallel and do not intersect
        x = None
        y = None
    elif slope1 is None:
        # First line is vertical
        x = intercept1
        y = slope2 * x + intercept2
    elif slope2 is None:
        # Second line is vertical
        x = intercept2
        y = slope1 * x + intercept1
    elif slope1 == 0:
        # First line is horizontal
        y = point1[1]
        x = (y - intercept2) / slope2
    elif slope2 == 0:
        # Second line is horizontal
        y = point3[1]
        x = (y - intercept1) / slope1        
    else:
        # Calculate the intersection point of the two lines
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1
    
    intersection_point = (x, y)

    return intersection_point


def check_line_intersection(start_point, end_point, table_sides, table_limits, offset=1):
    ''' Checks if intersection of two lines is within some given limits'''
    x_min, y_min, x_max, y_max = table_limits

    for idx, side in enumerate(table_sides):
        table_side_start = side[0]
        table_side_end = side[1]
        
        intersection_point = get_line_intersection(start_point, end_point, table_side_start, table_side_end)

        if intersection_point[0] is not None and intersection_point[1] is not None:
            x, y = intersection_point

            if (x_min - offset) <= x <= (x_max + offset) and (y_min - offset) <= y <= (y_max + offset):
                return intersection_point, side    
        
    return None, None


def reflect_line(point1, point2, point3, point4):
    '''Calculates the reflection of a line based on the intersection point of two given lines.
    It returns the reflected line projection
    '''
    # Calculate the intersection point of the two lines
    intersection_point = get_line_intersection(point1, point2, point3, point4)

    ray_line = (point1, point2)
    surface_line = (point3, point4)

    # Calculate the direction vector of the ray
    ray_dir = np.array(ray_line[1]) - np.array(ray_line[0])
    ray_start = np.array(ray_line[0])

    # Calculate the normal vector of the surface
    surface_dir = np.array(surface_line[1]) - np.array(surface_line[0])
    surface_normal = np.array([-surface_dir[1], surface_dir[0]])

    # Calculate the angle of incidence
    cos_angle = np.dot(ray_dir, surface_normal) / (np.linalg.norm(ray_dir) * np.linalg.norm(surface_normal))
    angle = np.arccos(cos_angle) * 180 / np.pi

    # Calculate the reflection line
    reflection_dir = ray_dir - 2 * np.dot(ray_dir, surface_normal) / np.dot(surface_normal, surface_normal) * surface_normal
    reflection_line = (intersection_point, intersection_point + reflection_dir*100)

    return reflection_line
