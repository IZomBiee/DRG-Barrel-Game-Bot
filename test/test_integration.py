import cv2
import keyboard
import numpy as np
import time

from drg_barrel_game_bot import HSVBasketDetector, TOMLSettingsLoader as TSL, KickManager, StateManager, PointBasketDetector \
, BorderManager

point_basket_detector = PointBasketDetector()
hsv_basket_detector = HSVBasketDetector()
border_manager = BorderManager(hsv_basket_detector)
kick_manager = KickManager()
state_manager = StateManager()

kick_waiting_time = 0
on_left_border_time = 0

video_path = r"C:\Users\patri\Desktop\just_basket.mkv"
video_fps = 60

video_reader = cv2.VideoCapture(video_path)
dt = 1/video_fps

velocity = 0
state_manager.state = 'Setup Borders'
while True:
    ret, frame = video_reader.read()
    x, y, w, h = TSL()["display"]["logic_resolution"]
    frame = frame[y:y+h, 0: w]

    match state_manager.state:
        case 'Setup Borders':
            border_manager.update_borders(frame)
            if state_manager.state_duration() > TSL()['basket']['border_setup_time']:
                state_manager.state = 'Waiting For Left Border'
        case 'Waiting For Left Border':
            if border_manager.is_on_left_border(frame):
                state_manager.state = 'On Left Border'
                point_basket_detector.point1[0] = border_manager.get_left_border()
        case 'On Left Border':
            on_left_border_time = time.perf_counter()
            if not border_manager.is_on_left_border(frame):
                state_manager.state = 'Calculating Kick Time'
            elif state_manager.state_duration() > 0.5:
                print("On Left Border too long!")
                state_manager.state = 'Waiting For Left Border'
        case 'Calculating Kick Time':
            if point_basket_detector.is_point2_correct_color(frame):
                velocity = point_basket_detector.distance/(time.perf_counter()-on_left_border_time)
                print(f"Velocity is {velocity}")
                state_manager.state = 'Waiting Time For Kick'
        case 'Waiting Time For Kick':
            if state_manager.state_duration() > kick_waiting_time:
                if kick_manager.can_kick(frame):
                    kick_manager.kick()
                    print("Kick!")
                else:
                    print("Can't Kick")
                state_manager.state = 'Waiting For Left Border'

    x = hsv_basket_detector.find_x(frame)
    if x is not None:
        frame = cv2.line(frame, (int(x+velocity//2), 0), (int(x+velocity//2), 300), (255, 0, 255), 3)
    frame = border_manager.draw_borders(frame)
    frame = point_basket_detector.draw_points(frame)

    cv2.imshow("Debug View", frame)
    cv2.waitKey(round(dt*1000))

