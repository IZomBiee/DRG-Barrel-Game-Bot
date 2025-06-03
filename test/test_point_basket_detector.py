import cv2
import time
from drg_barrel_game_bot import PointBasketDetector, TOMLSettingsLoader as TSL

video_path = r"C:\Users\patri\Desktop\just_basket.mkv"
video_fps = 60

video_reader = cv2.VideoCapture(video_path)

point1_time = time.perf_counter()
point1 = False

basket_detector = PointBasketDetector([300, 300], [800, 300], 15, [232, 255, 255], 200, 50)
while video_reader.isOpened():
    ret, frame = video_reader.read()
    x, y, w, h = TSL()["display"]["logic_resolution"]

    frame = frame[y:y+h, 0: w]
    if ret:
        draw_frame = frame

        if basket_detector.is_point1_correct_color(frame):
            point1_time = time.perf_counter()
            point1 = True
        if basket_detector.is_point2_correct_color(frame) and point1:
            print(f"Speed is {basket_detector.calculate_speed(time.perf_counter()-point1_time)} px/s")
            point1 = False
        draw_frame = basket_detector.draw_points(draw_frame)

        cv2.imshow("Video", draw_frame)
        cv2.waitKey(round(1/video_fps*1000))
    else:
        video_reader = cv2.VideoCapture(video_path)