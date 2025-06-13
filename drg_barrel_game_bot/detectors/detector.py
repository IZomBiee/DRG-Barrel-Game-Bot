import numpy as np

from abc import ABC
from ..toml_setting_loader import TOMLSettingLoader as TSL

class Detector(ABC):
    def find(self, image: np.ndarray) -> list[int] | None:
        raise NotImplementedError("Detector Method is Not Implemented")

