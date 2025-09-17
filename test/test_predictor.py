import unittest
import cv2
import json
import os

from drg_barrel_game_bot import Detector, Predictor

class TestIntegration(unittest.TestCase):
    def setUp(self):
        time_samples_path = r'test\test_samples\predictor'
        self.time_prediction_video_paths = os.listdir(time_samples_path)
        self.time_prediction_video_paths = map(lambda path: os.path.join(time_samples_path, path),
                               self.time_prediction_video_paths)

        self.border_detection_samples = []
        test_samples_path = r'test\test_samples\borders'
        for file in os.listdir(test_samples_path):
            name, extention = os.path.splitext(file)
            if extention == '.json':
                with open(os.path.join(test_samples_path, file)) as file:
                    data: dict = json.load(file)
                    for key in data.keys():
                        self.border_detection_samples.append({
                            'video_path':os.path.join(test_samples_path, key),
                            'data':data[key]
                            })

    def test_border_detection(self):
        for sample in self.border_detection_samples:
            path = sample['video_path']
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            detector = Detector()
            predictor = Predictor(detector)
            for frame in range(frame_count):
                predictor.update_borders(cap.read()[1])
            data = sample['data']
            self.assertAlmostEqual((predictor.right_border_x-predictor.left_border_x)*width, data[1]-data[0], delta=50, msg=f'"{path}" - border gap')

    def test_time_prediction(self):
        for path in self.time_prediction_video_paths:
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = round(frame_count/fps, 2)
            dt = 1/fps

            detector = Detector()
            predictor = Predictor(detector)
            for frame in range(frame_count):
                predictor.update_borders(cap.read()[1])
            predictor.on_direction_change()

            cap = cv2.VideoCapture(path)
            for i in range(frame_count):
                frame = cap.read()[1]
                predictor.update(frame, i*dt)

                if not predictor.is_on_setup_position():
                    cycle_time = predictor.cycle_time()
                    self.assertAlmostEqual(cycle_time, duration, delta=0.2, msg=f'"{path}" - {cycle_time}/{duration} on frame {i}')
                    break

if __name__ == '__main__':
    unittest.main()