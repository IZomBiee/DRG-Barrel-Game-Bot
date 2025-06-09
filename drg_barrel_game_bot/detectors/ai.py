import numpy as np
import cv2

from ..settings_loader import TOMLSettingsLoader as TSL
from .detector import Detector
from ultralytics import YOLO

class AIBasketDetector(Detector):
    '''Detector that uses YOLOv8 model to find baskets'''
    def __init__(self):
        settings = TSL()['detectors']['ai']
        model_path = settings['model_path']
        classes_path = settings['classes_path']
        self.score_threshold = settings['score_threshold']
        self.iou_threshold = settings['iou_threshold']
        self.target_size = settings['target_size']
        
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f]
        self.class_index = self.class_names.index('basket')  # Find basket class index
        
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
                    if int(box.cls) == self.class_index:  # Check if it's a basket
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2

                        return [round(center_x), round(center_y)]
        return None