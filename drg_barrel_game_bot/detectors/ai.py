import numpy as np

from drg_barrel_game_bot import TOMLSettingsLoader as TSL
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
        
        # Tracking parameters
        self.time_of_positions = settings['time_of_positions']
        self.positions = []
        self.delta_times = []
        self.avarage_velocity = [0., 0.]
        
        # Load class names
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f]
        self.class_index = self.class_names.index('basket')  # Find basket class index
        
        # Load YOLO model
        self.model = YOLO(model_path)

    def _add_position(self, position: list[int], dt: float) -> None:
        if sum(self.delta_times) > self.time_of_positions:
            self.positions.pop(0)
            self.delta_times.pop(0)
        self.positions.append(position)
        self.delta_times.append(dt)

    def _update_avarage_velocity(self) -> None:
        if len(self.positions) > 1:
            delta_positions_x = []
            delta_positions_y = []
            for i in range(len(self.positions)-1):
                delta_x = (self.positions[i+1][0] - self.positions[i][0]) / self.delta_times[i]
                delta_y = (self.positions[i+1][1] - self.positions[i][1]) / self.delta_times[i]
                delta_positions_x.append(delta_x)
                delta_positions_y.append(delta_y)
            
            if delta_positions_x:
                self.avarage_velocity[0] = sum(delta_positions_x) / len(delta_positions_x)
            if delta_positions_y:
                self.avarage_velocity[1] = sum(delta_positions_y) / len(delta_positions_y)

    def get_last_position(self) -> list[int] | None:
        return self.positions[-1] if self.positions else None

    def find(self, image: np.ndarray) -> list[int] | None:
        # Resize image to target size
        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.target_size, self.target_size))
        
        # Run YOLO inference
        results = self.model.predict(
            source=resized,
            conf=self.score_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls) == self.class_index:  # Check if it's a basket
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        
                        # Calculate center in resized coordinates
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        
                        # Convert coordinates to original image size
                        orig_x = int((center_x / self.target_size) * w)
                        orig_y = int((center_y / self.target_size) * h)
                        
                        return [orig_x, orig_y]
        return None

    def update(self, image: np.ndarray, dt: float) -> None:
        position = self.find(image)
        if position is not None:
            self._add_position(position, dt)
            self._update_avarage_velocity()
    
    def draw_basket_position(self, image: np.ndarray) -> np.ndarray:
        pos = self.get_last_position()
        if pos is not None:
            image = cv2.circle(image, pos, 15, (0, 255, 255), 3)
        return image

    def draw_trajectory(self, image: np.ndarray) -> np.ndarray:
        pos = self.get_last_position()
        if pos is not None:
            next_pos_x = int(self.avarage_velocity[0] + pos[0])
            next_pos_y = int(self.avarage_velocity[1] + pos[1])
            next_pos = [next_pos_x, next_pos_y]
            image = cv2.circle(image, next_pos, 15, (255, 0, 255), 3)
        return image