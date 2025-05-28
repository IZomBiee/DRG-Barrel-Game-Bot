import cv2
from drg_barrel_game_bot import DifferenceBasketDetector, HSVBasketDetector, TOMLSettingsLoader as TSL

TSL()

video_path = r"C:\Users\patri\Videos\2025-05-28 18-08-11.mkv"
video_fps = 60

video_reader = cv2.VideoCapture(video_path)


basket_detector = HSVBasketDetector()
while video_reader.isOpened():
    ret, frame = video_reader.read()
    if ret:
        draw_frame = frame
        x = basket_detector.find_x(frame)
        if x is not None:
            frame = cv2.line(draw_frame, (x, 0), (x, draw_frame.shape[0]), (255, 0, 255), 5)
        draw_frame = basket_detector.draw_borders(draw_frame)
        cv2.imshow("Video", draw_frame)
        cv2.waitKey(round(1/video_fps*1000))
    else:
        video_reader = cv2.VideoCapture(video_path)