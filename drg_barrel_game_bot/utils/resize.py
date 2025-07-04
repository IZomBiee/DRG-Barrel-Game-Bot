import cv2

class Resize:
    @staticmethod
    def letterbox(image, target_size=(640, 640), color=(114, 114, 114)):
        h, w = image.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                        borderType=cv2.BORDER_CONSTANT, value=color)

        return padded_image