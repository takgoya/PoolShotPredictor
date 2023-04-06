import cv2
import numpy as np

video_path = 'video\Shot-Predictor-Video.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

def dummy(val):
    pass

cv2.namedWindow('ColorPicker')
cv2.createTrackbar('hue_min', 'ColorPicker', 54, 179, dummy)
cv2.createTrackbar('hue_max', 'ColorPicker', 86, 179, dummy)
cv2.createTrackbar('sat_min', 'ColorPicker', 125, 255, dummy)
cv2.createTrackbar('sat_max', 'ColorPicker', 250, 255, dummy)
cv2.createTrackbar('val_min', 'ColorPicker', 92, 255, dummy)
cv2.createTrackbar('val_max', 'ColorPicker', 255, 255, dummy)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280,720))
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_min = cv2.getTrackbarPos('hue_min', 'ColorPicker')
    hue_max = cv2.getTrackbarPos('hue_max', 'ColorPicker')
    sat_min = cv2.getTrackbarPos('sat_min', 'ColorPicker')
    sat_max = cv2.getTrackbarPos('sat_max', 'ColorPicker')
    val_min = cv2.getTrackbarPos('val_min', 'ColorPicker')
    val_max = cv2.getTrackbarPos('val_max', 'ColorPicker')

    print(hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    # 54, 86, 125, 250, 92, 255

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    hsv_mask = cv2.inRange(frame_hsv, lower, upper)

    result_image = cv2.bitwise_and(frame, frame, mask=hsv_mask)

    cv2.imshow('HSV', result_image)

    if cv2.waitKey(fps) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()