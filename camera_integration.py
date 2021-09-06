import cv2
import numpy as np

res = '1080p'
vHeight = 1920
vWidth = 1080


def capture_settings(capture, height, width):
    capture.set(3, height)
    capture.set(4, width)


cap = cv2.VideoCapture(0)
capture_settings(cap, vHeight, vWidth)

while cap.isOpened():
    success, img = cap.read()

    cv2.imshow("Img", img)
    if cv2.waitKey(1) == ord('q'):
        break
