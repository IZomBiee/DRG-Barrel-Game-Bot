import cv2
import time

from drg_barrel_game_bot import BasketPredictor, AIBasketDetector, HSVBasketDetector, DifferenceBasketDetector


video_path = r"C:\Users\patri\Videos\2025-06-07 11-38-44.mkv"
video_fps = 30
dt = 1/video_fps

video_reader = cv2.VideoCapture(video_path)

frames_to_show_line = -1

detector = AIBasketDetector()
predictor = BasketPredictor(detector)
while video_reader.isOpened():
    ret, frame= video_reader.read()
    frame = AIBasketDetector.crop_to_logic_resolution(frame)
    if ret:
        draw_frame = frame.copy()
        
        predictor.update(frame, dt)
        predictor.update_borders()

        predictor.draw_basket(draw_frame)
        predictor.draw_trajectory(draw_frame)
        predictor.draw_borders(draw_frame)

        cv2.imshow("Draw Frame", draw_frame)
        cv2.waitKey(round(dt*1000))
    else:
        exit()