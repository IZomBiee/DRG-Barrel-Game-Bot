import numpy as np
import cv2

from .utils import Draw
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

        self.last_box = None

    def find(self, image: np.ndarray) -> list[float] | None:
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
                    self.last_box = box
                    x_min, y_min, x_max, y_max = box.xyxyn[0].tolist()
                    return [x_min, y_min, x_max, y_max]
        return None

    def draw(self, image: np.ndarray) -> np.ndarray:
        if self.last_box is None:
            return image
        
        height, width = image.shape[:2]

        x_min, y_min, x_max, y_max = self.last_box.xyxyn[0].tolist()
        x_min, y_min, x_max, y_max = [int(i) for i in (x_min*width, y_min*height, x_max*width, y_max*height)]
        center_x = (x_max+x_min)//2
        center_y = (y_max+y_min)//2
        confidence = float(self.last_box.conf)

        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        image = Draw.texts(image, center_x, y_min, [
            f'Conf:{round(confidence*100)}%',
            f'Pos:{center_x, center_y}'
        ], (255, 255, 255), direction=-1)
        
        return image
