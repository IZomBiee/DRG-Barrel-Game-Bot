import numpy as np

from abc import ABC
from ..settings_loader import TOMLSettingsLoader as TSL

class Detector(ABC):
    def find_x(self, image: np.ndarray) -> list[int] | None:
        raise NotImplementedError("Detector Method is Not Implemented")

    @staticmethod
    def crop_to_logic_resolution(image: np.ndarray) -> np.ndarray:
        x, y, w, h = TSL()["display"]["logic_resolution"]

        image = image[y:y+h, 0: w]
        return image
