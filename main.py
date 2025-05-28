import cv2
import keyboard
import numpy as np
import time

from drg_barrel_game_bot import BasketPredictor, Recorder

cam = Recorder(region=(0, 0, 1920, 1080), target_fps=30)
print("DRG Barrel Game Bot")
print("Press ctrl+p to start")
keyboard.wait("ctrl+p")

print("Grabing border information...")
start_time = time.perf_counter()
time_on_border = time.perf_counter()
show_speed = False

last_kick_time = time.perf_counter()

basket_predictor = BasketPredictor()
while True:
    frame = cam.get_last_frame()
    basket_predictor.update(frame)

    if time.perf_counter()-start_time < 3:
        continue

    if basket_predictor.on_left_border():
        print("On left Border!")
        time_on_border = time.perf_counter()
        show_speed = True

    if time.perf_counter()-time_on_border > 0.2 and show_speed:
        count = basket_predictor.frames_to_center()
        if time.perf_counter()-last_kick_time < 2.5:
            print("Can't shoot, barrel is in flaying")
        elif count is not None:
            sleep_time = count*(1/cam.target_fps)-2
            if sleep_time > 3:
                print("Error in calculation!")
            elif sleep_time > 0:
                print(f"Can fire on {sleep_time} seconds!")
                time.sleep(sleep_time)
                keyboard.press_and_release('e')
                last_kick_time = time.perf_counter()
            else:
                print("To Fast!")
        show_speed = False

    # frame = basket_predictor.draw_basket(frame)
    # frame = basket_predictor.draw_traectory(frame)
    # frame = basket_predictor.detector.draw_borders(frame)
    # cv2.imshow("Debug View", frame)
    # cv2.waitKey(1)