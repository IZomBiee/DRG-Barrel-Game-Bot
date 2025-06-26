import cv2
import numpy as np
import time
from drg_barrel_game_bot import Detector

test_video_path = r"C:\Users\patri\Videos\Timeline 1.mp4"

cap = cv2.VideoCapture(test_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = round(frame_count/fps, 2)
dt = 1/fps

detector = Detector()

cv2.imshow("Output", np.zeros_like((160, 160, 3), dtype=np.uint8))
cv2.waitKey(100)

while True:
    cap = cv2.VideoCapture(test_video_path)
    for i in range(frame_count):
        frame = cap.read()[1]

        pos = detector.find(frame)
        if pos is not None:
            detector.draw(frame)

        cv2.imshow("Output", frame)
        cv2.waitKey(10)

cv2.destroyAllWindows()
