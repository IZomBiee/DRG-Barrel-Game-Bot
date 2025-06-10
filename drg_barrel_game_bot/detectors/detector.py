import numpy as np

from abc import ABC
from ..toml_setting_loader import TOMLSettingLoader as TSL

class Detector(ABC):
    def find(self, image: np.ndarray) -> list[int] | None:
        raise NotImplementedError("Detector Method is Not Implemented")

    @staticmethod
    def crop_to_basket_y_gap(image: np.ndarray) -> np.ndarray:
        y0, y1= TSL()["display"]["basket_y_gap"]

        image = image[y0:y1]
        return image
