import unittest
import cv2
import json
import os

from drg_barrel_game_bot import KickManager

class TestKickManager(unittest.TestCase):
    def setUp(self):
        self.images = []
        self.values = []
        test_sample_path = r'test\test_samples\kick_manager'
        for file in os.listdir(test_sample_path):
            name, extention = os.path.splitext(file)
            if extention == '.json':
                with open(os.path.join(test_sample_path, file)) as file:
                    data: dict = json.load(file)
                    for key in data.keys():
                        self.images.append(cv2.imread(os.path.join(test_sample_path, key)))
                        self.values.append(data[key])

    def test_kick_manager(self):
        manager = KickManager()
        for image, value in zip(self.images, self.values):
            self.assertIs(manager.is_barrel_in_front(image), value)

if __name__ == '__main__':
    unittest.main()