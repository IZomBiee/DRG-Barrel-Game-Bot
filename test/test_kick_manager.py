import cv2
import time

from drg_barrel_game_bot import KickManager


video_path = r"C:\Users\patri\Videos\2025-05-28 18-08-11.mkv"
fps = 30
dt = 1/fps
kick_manager = KickManager()
video_reader = cv2.VideoCapture(video_path)

while video_reader.isOpened():
    ret, frame = video_reader.read()
    if ret:
        draw_frame = frame.copy()
        
        kick_manager.update(frame)

        kick_manager.draw_state(draw_frame)

        cv2.imshow("Video", draw_frame)
        cv2.waitKey(1)
    else:
        exit("Video Ended")