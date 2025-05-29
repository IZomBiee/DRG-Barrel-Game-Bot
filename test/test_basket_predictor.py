import cv2
import time

from drg_barrel_game_bot import BasketPredictor, HSVBasketDetector, TOMLSettingsLoader as TSL


video_path = r"C:\Users\patri\Videos\2025-05-29 14-01-34.mkv"
video_fps = 30

video_reader = cv2.VideoCapture(video_path)

frames_to_show_line = -1

basket_detector = HSVBasketDetector(500, TSL()['basket']['hsv_min'], TSL()['basket']['hsv_max'], [15, 15], 30, 1)
basket_predictor = BasketPredictor(basket_detector, 0.2, 30)
while video_reader.isOpened():
    ret, frame= video_reader.read()
    x, y, w, h = TSL()["display"]["resolution"]

    frame = frame[y:y+h, 0: w]
    if ret:
        basket_predictor.update(frame, 1/video_fps)
        frame = basket_detector._proccess_image(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        basket_predictor.detector.draw_borders(frame)
        basket_predictor.draw_basket(frame)
        basket_predictor.draw_traectory(frame)
        if basket_predictor.is_on_left_border():
            left_time = basket_predictor.time_to_right_border()
            if left_time is not None and left_time > 0:
                frames_to_show_line = round(left_time/(1/video_fps))
        
        print(f"Frames to position: {frames_to_show_line}")
        if frames_to_show_line >= 0:
            frames_to_show_line -= 1
        
        if frames_to_show_line == 0:
            cv2.waitKey(1000)

        cv2.imshow("Video", frame)
        cv2.waitKey(round(1/video_fps*1000))
    else:
        video_reader = cv2.VideoCapture(video_path)