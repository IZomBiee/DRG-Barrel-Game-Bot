import cv2
from drg_barrel_game_bot import HSVBasketDetector, TOMLSettingsLoader as TSL

video_path = r"C:\Users\patri\Desktop\just_basket.mkv"
video_fps = 60

video_reader = cv2.VideoCapture(video_path)

basket_detector = HSVBasketDetector(200, TSL()['basket']['hsv_min'], TSL()['basket']['hsv_max'], [15, 15], 30, 1)
while video_reader.isOpened():
    ret, frame = video_reader.read()
    x, y, w, h = TSL()["display"]["resolution"]

    frame = frame[y:y+h, 0: w]
    if ret:
        draw_frame = frame
        draw_frame = basket_detector._proccess_image(draw_frame)
        draw_frame = cv2.cvtColor(draw_frame, cv2.COLOR_GRAY2BGR)
        x = basket_detector.find_x(frame)
        if x is not None:
            frame = cv2.line(draw_frame, (x, 0), (x, draw_frame.shape[0]), (255, 0, 255), 5)
        draw_frame = basket_detector.draw_borders(draw_frame)

        cv2.imshow("Video", draw_frame)
        cv2.waitKey(round(1/video_fps*1000))
    else:
        video_reader = cv2.VideoCapture(video_path)