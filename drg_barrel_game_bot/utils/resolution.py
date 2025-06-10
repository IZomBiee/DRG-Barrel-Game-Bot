import numpy as np

def realative_resolution_to_absolute(image: np.ndarray,
                                     absolute_resolution: list[float]) -> list[int]:
    height, width, color_count = image.shape

    x0 = height//absolute_resolution[0]
    y0 = width//absolute_resolution[1]
    x1 = height//absolute_resolution[2]
    y1 = width//absolute_resolution[3]

    return x0, y0, x1, y1