import cv2
import numpy as np
import time
from drg_barrel_game_bot import Detector

test_video_path = r"test\test_samples\predictor\2025-05-29 14-01-34_00000397.mp4"

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
            x0, y0, x1, y1 = pos['box']
            frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 255), 1)
            frame = cv2.circle(frame, pos['center'], 5, (255, 255, 255), 1)
        else:
            print(f"NONE DETECTED! {time.perf_counter()}")
        cv2.imshow("Output", frame)
        cv2.waitKey(100)

cv2.destroyAllWindows()
