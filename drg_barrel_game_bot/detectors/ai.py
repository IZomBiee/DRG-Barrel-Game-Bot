import numpy as np
import cv2

from ..toml_setting_loader import TOMLSettingLoader as TSL
from .detector import Detector
from ultralytics import YOLO

class AIBasketDetector(Detector):
    '''Detector that uses YOLOv8 model to find basket'''
    def __init__(self):
        settings = TSL()['detectors']['ai']
        model_path = settings['model_path']
        self.score_threshold = settings['score_threshold']
        self.iou_threshold = settings['iou_threshold']
        self.target_size = settings['model_size']
        
        self.model = YOLO(model_path)

    def find(self, image: np.ndarray) -> list[int] | None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=image,
            imgsz=self.target_size,
            conf=self.score_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2

                    return [int(center_x), int(center_y)]
        return None