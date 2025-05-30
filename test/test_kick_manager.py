import cv2

from drg_barrel_game_bot import KickManager, TOMLSettingsLoader as TSL

kick_manager = KickManager(r"assets/e_button_3840x2160.png", [1920, 1080], 0.8, 0.4, 1)

video_path = r"C:\Users\patri\Videos\2025-05-28 18-08-11.mkv"
video_reader = cv2.VideoCapture(video_path)

while video_reader.isOpened():
    ret, frame = video_reader.read()
    x, y, w, h = TSL()["display"]["logic_resolution"]

    frame = frame[y:y+h, 0: w]
    if ret:
        draw_frame = frame
        
        if kick_manager.can_kick(frame):
            print("Kick")

        cv2.imshow("Video", draw_frame)
        cv2.waitKey(10)
    else:
        video_reader = cv2.VideoCapture(video_path)