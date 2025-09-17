import cv2
import keyboard

from drg_barrel_game_bot.utils import Draw
from drg_barrel_game_bot import Detector, Predictor

test_video_path = r"D:\Python\DRG-Barrel-Game-Bot\test\test_samples\predictor\2025-06-17 10-53-44_00000123.mov"
cap = cv2.VideoCapture(test_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
duration = round(frame_count/fps, 2)
dt = 1/fps

detector = Detector()
predictor = Predictor(detector)

for i in range(int(5/dt)):
    print(f"Border detection {i}/{int(5/dt)} frames")
    frame = cap.read()[1]
    if frame is None:
        break
    predictor.update_borders(frame)
    print("\033[A                             \033[A")

predictor.on_direction_change()

processed_frame_buffer = []
position_buffer = []

cap = cv2.VideoCapture(test_video_path)
for i in range(frame_count):
    print(f"Processed {i}/{frame_count} frames")
    frame = cap.read()[1]
    if frame is None:
        break
    predictor.update(frame, i*dt)
    if detector.last_box is not None:
        position_buffer.append(detector.last_box.xyxy.tolist()[0])
    else: 
        position_buffer.append(None)

    frame = predictor.draw(frame)
    frame = predictor.draw_trail(frame, 5)
    frame = Draw.text(frame, 200, 200, f'Frame: {i+1}/{frame_count}', (255, 255, 255))
    processed_frame_buffer.append(frame)
    print("\033[A                             \033[A")

for i in range(frame_count):
    print(f"Next pos marked {i}/{frame_count} frames")

    frame = processed_frame_buffer[i]
    height, width = frame.shape[:2]
    correct_pos = int(i+(1/dt))

    if correct_pos < frame_count:
        if position_buffer[correct_pos] is not None:
            x0, y0, x1, y1 = position_buffer[correct_pos]
            frame = cv2.rectangle(frame, (int(x0), int(y0)), 
                            (int(x1), int(y1)), (255, 0, 255), 3)
            frame = cv2.circle(frame, (int((x0+x1)/2), int((y0+y1)/2)), 
                            5, (255, 255, 255), 3)
            
    processed_frame_buffer.append(frame)
    print("\033[A                             \033[A")
    
current_frame = 0
while True:
    if keyboard.is_pressed('a') and current_frame > 0:current_frame -= 1
    elif keyboard.is_pressed('d') and current_frame < frame_count-1:current_frame += 1
    cv2.imshow("Video", processed_frame_buffer[current_frame])
    cv2.waitKey(int(dt*1000))
