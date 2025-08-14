import cv2
import keyboard
import time
import torch

from collections import deque
from drg_barrel_game_bot import Detector, \
WindowRecorder, SL, KickManager, StateManager, Predictor
from drg_barrel_game_bot.utils import Resize

if torch.cuda.is_available():
    print("CUDA detected, GPU acceleration will be used.")
else: print("No CUDA detected, using cpu!")

cam = WindowRecorder()

detector = Detector()
kick_manager = KickManager()
state_manager = StateManager()
predictor = Predictor(detector)

print(f"\nPRESS {SL()['program']['start_key'].upper()} TO START AND {SL()['program']['stop_key'].upper()} TO STOP. PREPARE CORRECT POSITION!")

kick_waiting_time = 0
on_left_border_time = 0
last_start_key_time = 0

video_writer = None
if SL()['display']['debug_video']:
    debug_video_path = 'debug_view.mp4'
    print(f"Starting writing video at {debug_video_path}")
    video_writer = cv2.VideoWriter(debug_video_path, cv2.VideoWriter.fourcc(*'avc1'),
                                   20.0, SL()['display']['debug_view_resolution'])

fps_list = deque(maxlen=20)
while True:
    loop_start_time = time.perf_counter()
    frame = cam.get_frame()
    if frame is None or not cam.is_updated():continue

    predictor.update(frame, time.perf_counter())
    kick_manager.update(frame)
    match state_manager.state:
        case "On Startup":
            if keyboard.is_pressed(SL()['program']['start_key']): 
                state_manager.state = 'Setup Borders'
                last_start_key_time = time.perf_counter()
            cam.update_region()
        case 'Setup Borders':
            predictor.update_borders()
            if state_manager.state_duration() > SL()['basket_predictor']['border_setup_time']:
                state_manager.state = 'Waiting For Left Border'
        case 'Waiting For Left Border':
            if predictor.on_left_border():
                state_manager.state = 'On Left Border'
        case 'On Left Border':
            on_left_border_time = time.perf_counter()
            if not predictor.on_left_border():
                state_manager.state = 'Calculating Kick Time'
            elif state_manager.state_duration() > 0.5:
                print("On Left Border too long!")
                state_manager.state = 'Waiting For Left Border'
        case 'Calculating Kick Time':
            if not predictor.is_on_setup_position():
                delay = predictor.time_to_right_border()+predictor.cycle_time()/2
                delay -= SL()['basket_predictor']['barrel_fly_time']
                print(f"Time to left border:{delay}")
                if delay > 0:
                    kick_waiting_time = delay
                    state_manager.state = 'Waiting Time For Kick'
                else:
                    state_manager.state = 'Waiting For Left Border'
        case 'Waiting Time For Kick':
            if state_manager.state_duration() > kick_waiting_time:
                if kick_manager.can_kick(frame):
                    kick_manager.kick()
                state_manager.state = 'Waiting For Left Border'

    if SL()['display']['debug_view'] or video_writer is not None:
        frame = predictor.draw(frame)
        
        frame = Resize.letterbox(frame, SL()['display']['debug_view_resolution'], (0, 0, 0))
        frame = kick_manager.draw_state(frame)
        fps = 1//(time.perf_counter()-loop_start_time)
        fps_list.append(fps)
        frame = cv2.putText(frame, f'FPS: {sum(fps_list)//len(fps_list)}',
                            (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2)

        if SL()['display']['debug_view']:
            cv2.imshow("Debug View", frame)
            cv2.imshow("Debug View", frame)
            cv2.waitKey(1)
        
        if video_writer is not None:
            video_writer.write(frame)
    
    if keyboard.is_pressed(SL()['program']['stop_key']) and \
        time.perf_counter()-last_start_key_time>1 and \
        state_manager.state != "On Startup":
        break

if video_writer is not None:
    print("Saving debug video...")
    video_writer.release() 
    cv2.destroyAllWindows()

print("GOODBYE!")