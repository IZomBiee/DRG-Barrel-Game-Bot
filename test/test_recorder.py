from drg_barrel_game_bot import TSL, WindowRecorder
import cv2
recorder = WindowRecorder()

while True:
    recorder.update_region()
    cv2.imshow("Recorder", recorder.get_frame())
    cv2.waitKey(1)