import cv2
from drg_barrel_game_bot import AIBasketDetector, TOMLSettingsLoader as TSL

video_path = r"C:\Users\patri\Videos\2025-06-07 11-38-16.mkv"
dt = 1/30

video_reader = cv2.VideoCapture(video_path)

hsv_basket_detector = AIBasketDetector()
while video_reader.isOpened():
    ret, frame = video_reader.read()
    # frame = hsv_basket_detector.crop_to_logic_resolution(frame)
    if ret:
        hsv_basket_detector.update(frame,  dt)
        draw_frame = frame.copy()
        draw_frame = hsv_basket_detector.draw_basket_position(draw_frame)
        draw_frame = hsv_basket_detector.draw_trajectory(draw_frame)
        
        cv2.putText(draw_frame, f"Vel: {hsv_basket_detector.avarage_velocity[0]}", (0, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0))
        
        cv2.imshow("Video", draw_frame)
        cv2.waitKey(1)
    else:
        video_reader = cv2.VideoCapture(video_path)