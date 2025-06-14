import cv2
import keyboard
import numpy as np
import time
import torch

from drg_barrel_game_bot import AIBasketDetector, \
WindowRecorder, TSL, KickManager, StateManager, BasketPredictor

print(r"""
________ __________  ________  __________                             .__   
\______ \\______   \/  _____/  \______   \_____ ______________   ____ |  |  
 |    |  \|       _/   \  ___   |    |  _/\__  \\_  __ \_  __ \_/ __ \|  |  
 |    `   \    |   \    \_\  \  |    |   \ / __ \|  | \/|  | \/\  ___/|  |__
/_______  /____|_  /\______  /  |______  /(____  /__|   |__|    \___  >____/
        \/       \/        \/          \/      \/                   \/      
  ________                        __________        __                      
 /  _____/_____    _____   ____   \______   \ _____/  |_                    
/   \  ___\__  \  /     \_/ __ \   |    |  _//  _ \   __\                   
\    \_\  \/ __ \|  Y Y  \  ___/   |    |   (  <_> )  |                     
 \______  (____  /__|_|  /\___  >  |______  /\____/|__|                     
        \/     \/      \/     \/          \/                                
""")
print(f"Loaded this settings:\n{TSL()}\n")

if torch.cuda.is_available():
    print("CUDA detected, GPU acceleration will be used")
else: print("No CUDA detected, using cpu!")

cam = WindowRecorder()

detector = AIBasketDetector()
kick_manager = KickManager()
state_manager = StateManager()
predictor = BasketPredictor(detector)

print(f"\nPRESS {TSL()['program']['start_key'].upper()} WHEN YOU ON THE RIGHT SPOT")

kick_waiting_time = 0
on_left_border_time = 0

update_time = time.perf_counter()
while True:
    dt = time.perf_counter()-update_time
    update_time = time.perf_counter()

    frame = cam.get_frame()

    predictor.update(frame, dt)
    kick_manager.update(frame)
    match state_manager.state:
        case "On Startup":
            if keyboard.is_pressed(TSL()['program']['start_key']): 
                state_manager.state = 'Setup Borders'
            cam.update_region()
        case 'Setup Borders':
            predictor.update_borders()
            if state_manager.state_duration() > TSL()['basket_predictor']['border_setup_time']:
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
            if state_manager.state_duration() > TSL()['basket_predictor']['velocity_checking_time']:
                delay = predictor.time_to_left_border()+predictor.cycle_time()/2
                delay -= TSL()['basket_predictor']['barrel_fly_time']
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

    if TSL()['display']['debug_view']:
        frame = predictor.draw_basket(frame)
        frame = predictor.draw_trajectory(frame)
        frame = predictor.draw_borders(frame)
        frame = kick_manager.draw_state(frame)
        

        cv2.imshow("Debug View", frame)
        # cv2.imshow("Debug View", cv2.resize(frame, TSL()['display']['debug_view_resolution']))
        cv2.waitKey(1)

