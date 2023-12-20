import cv2
import time

import numpy as np

vidcap = cv2.VideoCapture("cars_freeway(360p).mp4")
success, image = vidcap.read()

#setting FPS
frame_rate = 30.0
delay = 1/frame_rate

def nothing(x):
    pass
#creating TrackBars
cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
while success:

    frame = cv2.resize(image, (640, 480))

    tl = (191, 311)
    tr = (413, 310)
    bl = (17, 444)
    br = (590, 447)

    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    pts1 = np.float32((tl, tr, bl, br))
    pts2 = np.float32(([0, 0], [640, 0], [0, 480], [640, 480]))

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    #Image Thresholding
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # Histogram
    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = np.int_(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y > 0:
        ## Left threshold
        img = mask[y - 40:y, left_base - 50:left_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        ## Right threshold
        img = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40

    #Display frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Transformed Frame Birds eye view", transformed_frame)
    cv2.imshow("Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    if cv2.waitKey(1) == 27:
        break
    time.sleep(delay)

    success, image = vidcap.read()

cv2.release()
cv2.destroyAllWindows()
