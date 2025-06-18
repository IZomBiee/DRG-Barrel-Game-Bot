import numpy as np
import cv2

from .setting_loader import SettingLoader as SL
from ultralytics import YOLO

class Detector:
    '''Detector that uses YOLOv8 model to find basket'''
    def __init__(self):
        settings = SL()['detectors']['ai']
        model_path = settings['model_path']
        self.score_threshold = settings['score_threshold']
        self.iou_threshold = settings['iou_threshold']
        self.target_size = settings['model_size']
        self.model = YOLO(model_path)

    def find(self, image: np.ndarray) -> dict | None:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        results = self.model.predict(
            source=image,
            imgsz=self.target_size,
            conf=self.score_threshold,
            iou=self.iou_threshold,
            save=False,
            save_txt=False,
            save_conf=False,
            verbose=False,
            device=''
        )

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x_min, y_min, x_max, y_max = list(map(lambda x: int(x), box.xyxy[0].tolist()))
                    center_x = int((x_min + x_max) / 2)
                    center_y = int((y_min + y_max) / 2)
                    return {
                        'center': [center_x, center_y],
                        'left': [x_min,center_y],
                        'right': [x_max,center_y],
                        'box': [x_min, y_min, x_max, y_max]
                    }
        return None