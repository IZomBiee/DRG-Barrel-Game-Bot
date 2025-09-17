import cv2
import keyboard
import time
import torch

from drg_barrel_game_bot.utils.draw import Draw
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

fps_list = deque(maxlen=20)
while True:
    loop_start_time = time.perf_counter()
    frame = cam.get_frame()
    if frame is None or not cam.is_updated():continue

    fps_len = len(fps_list)
    if fps_len == 0:
        avg_fps = 0
    else:
        avg_fps = sum(fps_list)//fps_len

    predictor.update(frame, time.perf_counter())

    kick_manager.update(frame)
    match state_manager.state:
        case "On Startup":
            if keyboard.is_pressed(SL()['program']['start_key']): 
                if SL()['display']['debug_video']:
                    debug_video_path = 'debug_view.mp4'
                    print(f"Starting writing video at {debug_video_path}")
                    video_writer = cv2.VideoWriter(debug_video_path,
                        cv2.VideoWriter.fourcc(*'avc1'),
                        avg_fps, (cam.region['width'], cam.region['height']))

                state_manager.state = 'Setup Borders'
                last_start_key_time = time.perf_counter()
            cam.update_region()
        case 'Setup Borders':
            predictor.update_borders()
            if state_manager.state_duration() > SL()['basket_predictor']['border_setup_time']:
                state_manager.state = 'Waiting For Left Border'
        case 'Waiting For Left Border':
            if predictor.is_moving_right() and predictor.is_on_left_border():
                state_manager.state = 'Calculating Kick Time'
        case 'Calculating Kick Time':
            if not predictor.is_on_setup_position():
                delay = predictor.time_to_right_border()+predictor.cycle_time()/2
                if delay > SL()['basket_predictor']['max_time']:
                    print(f"The waiting time is too big! ({delay})")
                    state_manager.state = 'Waiting For Left Border'
                    continue
                
                delay -= SL()['basket_predictor']['barrel_fly_time']
                
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
        
        frame = kick_manager.draw_state(frame)
        frame = Draw.texts(
            frame, 180, frame.shape[0], [
                f'FPS: {avg_fps}',
                f'State: {state_manager.state}',
                f'Moving: {"Right" if predictor.is_moving_right() else "Left"}',
                f'On Left Border: {predictor.is_on_left_border()}'
            ], (255, 255, 255), -1
        )
        
        height, width = frame.shape[:2]
        scalling = SL()['display']['debug_view_scalling']
        frame = cv2.resize(frame, (int(width*scalling),
                                   int(height*scalling)))

        if SL()['display']['debug_view']:
            cv2.imshow("Debug View", frame)
            cv2.waitKey(1)
        
        if video_writer is not None:
            video_writer.write(frame)
    
    fps = 1//(time.perf_counter()-loop_start_time)
    fps_list.append(fps)

    if keyboard.is_pressed(SL()['program']['stop_key']) and \
        time.perf_counter()-last_start_key_time>1 and \
        state_manager.state != "On Startup":
        break

if video_writer is not None:
    print("Saving debug video...")
    video_writer.release() 
    cv2.destroyAllWindows()

print("GOODBYE!")