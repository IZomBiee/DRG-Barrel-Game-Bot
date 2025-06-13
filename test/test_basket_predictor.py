import cv2

from drg_barrel_game_bot import BasketPredictor, AIBasketDetector, HSVBasketDetector, DifferenceBasketDetector


video_path = r"C:\Users\patri\Videos\2025-06-07 11-38-44.mkv"
video_fps = 30
dt = 1/video_fps

video_reader = cv2.VideoCapture(video_path)

detector = AIBasketDetector()
predictor = BasketPredictor(detector)
while video_reader.isOpened():
    ret, frame= video_reader.read()
    if ret:
        draw_frame = frame.copy()
        
        predictor.update(frame, dt)
        predictor.update_borders()

        predictor.draw_basket(draw_frame)
        predictor.draw_trajectory(draw_frame)
        predictor.draw_borders(draw_frame)

        cv2.putText(draw_frame, f"Time To Left {round(predictor.time_to_left_border(), 2)}", (0, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        cv2.putText(draw_frame, f"Cycle Time {round(predictor.cycle_time(), 2)}", (0, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        cv2.imshow("Draw Frame", draw_frame)
        # cv2.waitKey(round(dt*1000))
        cv2.waitKey(1)
    else:
        exit()