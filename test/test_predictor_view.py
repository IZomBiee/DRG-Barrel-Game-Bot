import cv2

from drg_barrel_game_bot import Detector, Predictor

test_video_path = r"C:\Users\patri\Videos\Timeline 1.mp4"

cap = cv2.VideoCapture(test_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = round(frame_count/fps, 2)
dt = 1/fps

detector = Detector()
predictor = Predictor(detector)
for _ in range(frame_count):
    predictor.update_borders(cap.read()[1])
predictor.clear()

cap = cv2.VideoCapture(test_video_path)
for i in range(frame_count):
    frame = cap.read()[1]
    predictor.update(frame, i*dt)
    predictor.draw(frame)

    cv2.imshow("Output", frame)
    cv2.waitKey(50)
cv2.destroyAllWindows()
