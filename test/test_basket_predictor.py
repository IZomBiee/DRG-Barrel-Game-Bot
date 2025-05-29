import cv2
import time

from drg_barrel_game_bot import BasketPredictor, HSVBasketDetector, TOMLSettingsLoader as TSL


video_path = r"C:\Users\patri\Desktop\just_basket.mkv"
video_fps = 30

video_reader = cv2.VideoCapture(video_path)


basket_detector = HSVBasketDetector(200, TSL()['basket']['hsv_min'], TSL()['basket']['hsv_max'], [15, 15], 30, 1)
basket_predictor = BasketPredictor(basket_detector, 0.2, 30)
while video_reader.isOpened():
    ret, frame= video_reader.read()
    x, y, w, h = TSL()["display"]["resolution"]

    frame = frame[y:y+h, 0: w]
    if ret:
        basket_predictor.update(frame, 1/video_fps)
        basket_predictor.detector.draw_borders(frame)
        basket_predictor.draw_basket(frame)
        basket_predictor.draw_traectory(frame)
        print(basket_predictor.cycle_time())
        cv2.imshow("Video", frame)
        cv2.waitKey(round(1/video_fps*1000))
    else:
        video_reader = cv2.VideoCapture(video_path)