import cv2

from drg_barrel_game_bot import Detector, Predictor

test_video_path = r"test\test_samples\predictor\2025-06-17 10-53-44_00000552.mov"

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
    predictor.draw_basket(frame)
    predictor.draw_borders(frame)
    predictor.draw_trajectory(frame)
    cv2.putText(frame, f'Cycle Time: {predictor.cycle_time()}/{duration}', (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, f'Calculated Speed: {int(predictor.avarage_velocity[0])} px/s', (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, f'Correct Speed: {int((predictor.right_border_x-predictor.left_border_x)/duration)*2} px/s',
                (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, f'Borders: {predictor.left_border_x}-{predictor.right_border_x}', (20, 320),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, f'Setup Position: {predictor.is_on_setup_position()}', (20, 420),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.putText(frame, f'Frame: {i}', (20, 520),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow("Output", frame)
    cv2.waitKey(150)
cv2.destroyAllWindows()
