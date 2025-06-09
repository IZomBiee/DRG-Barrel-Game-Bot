import cv2

from drg_barrel_game_bot import KickManager

kick_manager = KickManager()

video_path = r"C:\Users\patri\Videos\2025-05-28 18-08-11.mkv"
video_reader = cv2.VideoCapture(video_path)

while video_reader.isOpened():
    ret, frame = video_reader.read()
    if ret:
        draw_frame = frame.copy()
        
        if kick_manager.can_kick(frame):
            text = "Can Kick"
            color = (0, 255, 0)
        else:
            text = "Can't Kick"
            color = (0, 0, 255)
            
        cv2.putText(draw_frame, text, (0, 150), cv2.FONT_HERSHEY_COMPLEX, 3, color, 3)

        cv2.imshow("Video", draw_frame)
        cv2.waitKey(1)
    else:
        exit("Video Ended")