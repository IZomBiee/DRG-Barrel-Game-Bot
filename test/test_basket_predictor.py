import cv2
import time

from drg_barrel_game_bot import BasketPredictor, TOMLSettingsLoader as TSL


video_path = r"C:\Users\patri\Desktop\just_basket.mkv"
video_fps = 30

video_reader = cv2.VideoCapture(video_path)



basket_predictor = BasketPredictor()
while video_reader.isOpened():
    ret, frame= video_reader.read()
    x, y, w, h = TSL()["display"]["resolution"]

    frame = frame[y:y+h, 0: w]
    if ret:
        basket_predictor.update(frame, 1/video_fps)
        basket_predictor.detector.draw_borders(frame)
        basket_predictor.draw_basket(frame)
        basket_predictor.draw_traectory(frame)

        cv2.imshow("Video", frame)
        cv2.waitKey(round(1/video_fps*1000))
    else:
        video_reader = cv2.VideoCapture(video_path)