import cv2
from drg_barrel_game_bot import DifferenceBasketDetector, TOMLSettingsLoader as TSL

video_path = r"C:\Users\patri\Videos\2025-06-07 11-38-16.mkv"
dt = 20

video_reader = cv2.VideoCapture(video_path)

diff_basket_detector = DifferenceBasketDetector()
while video_reader.isOpened():
    ret, frame = video_reader.read()
    frame = diff_basket_detector.crop_to_logic_resolution(frame)
    if ret:
        cv2.imshow("Process View", diff_basket_detector._process_image(frame, rewrite_last_image=False))
        diff_basket_detector.update(frame, dt/1000)
        draw_frame = frame.copy()
        draw_frame = diff_basket_detector.draw_basket_position(draw_frame)
        draw_frame = diff_basket_detector.draw_traectory(draw_frame)
        
        cv2.putText(draw_frame, f"Vel: {diff_basket_detector.avarage_velocity[0]}", (0, 200),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0))
        

        cv2.imshow("Video", draw_frame)
        cv2.waitKey(dt)
    else:
        video_reader = cv2.VideoCapture(video_path)