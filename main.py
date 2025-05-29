import cv2
import keyboard
import numpy as np
import time

from drg_barrel_game_bot import BasketPredictor, Recorder, TOMLSettingsLoader as TSL

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
print(f"Loaded this settings:\n{TSL()}")

print(f"\nPRESS {TSL()['program']['start_key'].upper()} WHEN YOU ON THE RIGHT SPOT")

cam = Recorder(region=TSL()['display']['resolution'])

while not keyboard.is_pressed(TSL()['program']['start_key']):
    if TSL()['display']['debug_view']:
        cv2.imshow("Debug View", cam.get_screenshot())
        cv2.waitKey(1)
    time.sleep(0.01)

print("Grabing border information...")
start_time = time.perf_counter()
time_on_border = time.perf_counter()
show_speed = False

last_kick_time = time.perf_counter()

basket_predictor = BasketPredictor()

update_time = time.perf_counter()
while True:
    dt = time.perf_counter()-update_time
    update_time = time.perf_counter()

    frame = cam.get_screenshot()
    basket_predictor.update(frame, dt)

    if time.perf_counter()-start_time < TSL()['basket']['border_setup_time']:
        pass
    else:
        if basket_predictor.on_left_border():
            print("On left Border!")
            time_on_border = time.perf_counter()
            show_speed = True

        if time.perf_counter()-time_on_border > TSL()['basket']['velocity_checking_time'] and show_speed:
            count = basket_predictor.time_to_right_border()
            cycle_time = basket_predictor.cycle_time()
            if time.perf_counter()-last_kick_time < TSL()['barrel']['respawn_time']:
                print("Can't shoot, barrel is in flaying")
            elif count is not None and cycle_time is not None:
                sleep_time = (count+cycle_time//2)-TSL()['barrel']['fly_time']
                if sleep_time > TSL()['kick']['maximal_time']:
                    print("Too big kick time!")
                elif sleep_time < TSL()['kick']['minimal_time']:
                    print("Too small kick time!")
                elif sleep_time > 0:
                    print(f"Can fire on {sleep_time} seconds!")
                    time.sleep(sleep_time)
                    keyboard.press_and_release('e')
                    last_kick_time = time.perf_counter()

            show_speed = False

    if TSL()['display']['debug_view']:
        frame = basket_predictor.draw_basket(frame)
        frame = basket_predictor.draw_traectory(frame)
        frame = basket_predictor.detector.draw_borders(frame)
        cv2.imshow("Debug View", frame)
        cv2.imshow("Proccess View", basket_predictor.detector._proccess_image(frame))
        cv2.waitKey(1)

    # print(f"FPS: {1/(time.perf_counter()-update_time)}")
