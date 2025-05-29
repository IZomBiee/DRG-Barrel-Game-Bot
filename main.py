import cv2
import keyboard
import numpy as np
import time

from drg_barrel_game_bot import BasketPredictor, HSVBasketDetector, \
Recorder, TOMLSettingsLoader as TSL, KickManager

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

cam = Recorder(region=TSL()['display']['logic_resolution'])

while not keyboard.is_pressed(TSL()['program']['start_key']):
    if TSL()['display']['debug_view']:
        cv2.imshow("Debug View", cam.get_screenshot())
        cv2.waitKey(1)
    time.sleep(0.01)

basket_detector = HSVBasketDetector(TSL()['basket']['min_area'], TSL()['basket']['hsv_min'],
                                    TSL()['basket']['hsv_max'], [15, 15], 30, 1)
basket_predictor = BasketPredictor(basket_detector, TSL()['basket']['position_count'],
                                   TSL()['basket']['border_tolirance'])
kick_manager = KickManager(r"assets/kick_label_3840x2160.png",TSL()['display']['resolution'],
                           TSL()['kick']['kicking_detection_sensitivity'], TSL()['kick']['barrel_boucing_time'],
                           TSL()['kick']['minimal_kick_delay'])

update_time = time.perf_counter()
while True:
    dt = time.perf_counter()-update_time
    update_time = time.perf_counter()

    frame = cam.get_screenshot()
    basket_predictor.update(frame, dt)

    if basket_predictor.is_on_left_border():
        print("Basket is on left border")
        if basket_predictor.avarage_velocity_x > 150:
            print("Velocity is positive")
            if kick_manager.can_kick(frame):
                time_to_kick = basket_predictor.time_to_right_border()-TSL()['barrel']['fly_time']
                print(f"Kick on {round(time_to_kick, 2)} s")

    if TSL()['display']['debug_view']:
        frame = basket_predictor.draw_basket(frame)
        frame = basket_predictor.draw_traectory(frame)
        frame = basket_predictor.detector.draw_borders(frame)

        cv2.imshow("Debug View", frame)
        cv2.imshow("Proccess View", basket_predictor.detector._proccess_image(frame))
        cv2.waitKey(1)

