import numpy as np
import cv2

from .utils import Draw
from .setting_loader import SettingLoader as SL
from yolov8_onnx import DetectEngine

class Detector:
    '''Detector that uses YOLOv8 model to find basket'''
    def __init__(self):
        settings = SL()['detectors']['ai']
        model_path = settings['model_path']
        self.score_threshold = settings['score_threshold']
        self.iou_threshold = settings['iou_threshold']
        self.target_size = settings['model_size']
        self.engine = DetectEngine(
            model_path,
            self.target_size,
            self.score_threshold,
            self.iou_threshold
        )
        
        self.last_results = []
        print("Initialized Detector with "
              f"target size {self.target_size} and model from "
              f"{model_path}.")

    def letterbox(self, image: np.ndarray,
                size: tuple[int, int], fill_value: int = 114):
        original_shape = image.shape[:2]
        r = min(size[0] / original_shape[0], size[1] / original_shape[1])
        new_unpadded = (int(round(original_shape[1] * r)), int(round(original_shape[0] * r)))

        resized = cv2.resize(image, new_unpadded, interpolation=cv2.INTER_LINEAR)
        dw = size[1] - new_unpadded[0]
        dh = size[0] - new_unpadded[1]
        dw /= 2
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(fill_value,) * 3)

        return padded, r, (dw, dh)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        self.original_shape = image.shape[:2]
        image, self.scale, self.pad = self.letterbox(image, (self.target_size, self.target_size))
        return image

    def postprocess(self, output) -> list[dict]:
        height, width = self.original_shape
        scale = self.scale
        pad_w, pad_h = self.pad
        results = []
        
        for i in range(0, len(output), 3):
            x0, y0, x1, y1 = output[i][0]
            conf = float(output[i + 1][0])
            cls = int(output[i + 2][0])

            x0 = max((x0 - pad_w) / scale, 0)
            y0 = max((y0 - pad_h) / scale, 0)
            x1 = min((x1 - pad_w) / scale, width)
            y1 = min((y1 - pad_h) / scale, height)

            results.append({
                'box':(int(x0), int(y0), int(x1), int(y1)),
                'normalized_box':(x0/width, y0/height, x1/width, y1/height),
                'conf': conf,
                'class': 'basket'
                })
        self.last_results = results
        return results

    def detect(self, image: np.ndarray) -> list[dict]:
        preprocessed = self.preprocess(image)
        output = self.engine(preprocessed)
        if output is None or len(output[0]) < 1:
            return []
        results = self.postprocess(output)
        self.last_results = results
        return results

    def draw(self, image: np.ndarray) -> np.ndarray:
        for result in self.last_results:
            x0, y0, x1, y1 = result['box']
            conf, cls_id = result['conf'], result['class']
            label = f"{cls_id}: {conf:.2f}"

            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
            Draw.text(image, (x0+x1)/2, y0, label, (255, 255, 255))
        return image