import cv2
import numpy as np
import keyboard
from drg_barrel_game_bot import Detector

test_video_path = r"test\test_samples\predictor\2025-06-17 10-53-44_00000123.mov"

cap = cv2.VideoCapture(test_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = round(frame_count/fps, 2)
dt = 1/fps

detector = Detector()

processed_frame_buffer = []

cap = cv2.VideoCapture(test_video_path)
for i in range(frame_count):
    print(f"Processed {i}/{frame_count} frames")
    frame = cap.read()[1]

    pos = detector.find(frame)
    if pos is not None:
        frame = detector.draw(frame)
    processed_frame_buffer.append(frame)
    print("\033[A                             \033[A")
    
current_frame = 0
while True:
    if keyboard.is_pressed('a') and current_frame > 0:current_frame -= 1
    elif keyboard.is_pressed('d') and current_frame < frame_count-1:current_frame += 1
    frame = processed_frame_buffer[current_frame]

    cv2.imshow("Video", frame)
    cv2.waitKey(int(dt*1000))

