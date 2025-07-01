import cv2
import keyboard

from drg_barrel_game_bot import Detector, Predictor

test_video_path = r"test\test_samples\predictor\2025-06-17 10-53-44_00000123.mov"

cap = cv2.VideoCapture(test_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = round(frame_count/fps, 2)
dt = 1/fps

detector = Detector()
predictor = Predictor(detector)
for _ in range(frame_count):
    predictor.update_borders(cap.read()[1])
predictor.clear()

processed_frame_buffer = []

cap = cv2.VideoCapture(test_video_path)
for i in range(frame_count):
    frame = cap.read()[1]
    frame = cv2.resize(frame, (800, 800))
    predictor.update(frame, i*dt)
    frame = predictor.draw(frame)
    print(f"Marking {i} frame...")
    processed_frame_buffer.append(frame)

current_frame = 0
while True:
    if keyboard.is_pressed('a'):current_frame -= 1
    elif keyboard.is_pressed('d'):current_frame += 1
    cv2.imshow("Video", processed_frame_buffer[current_frame])
    cv2.waitKey(int(dt*1000))
