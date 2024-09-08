import os
import cv2
import logging
import numpy as np

LOG = logging.getLogger("Quantization DataLoader:")

class RandomLoader(object):
    def __init__(self, target_size):
        self.target_size = target_size
        LOG.warning("Generate quantization data from random, it's will lead to accuracy problem!")
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index > 5:
            raise StopIteration()
        self.index += 1
        return [np.random.randn(*self.target_size).astype(np.float32)]
    
class ImageLoader(object):
    '''
        Generate data for quantization from image datas.
        img_quan_data = (img - mean) / std, it's important for accuracy of model.
    '''
    VALID_FORMAT = ['.jpg', '.png', '.jpeg']
    
    def __init__(self, img_root, target_size, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) -> None:
        assert os.path.exists(img_root), f"{img_root} does not exist, please check!"
        self.fns = self._load_img_paths(img_root)
        self.nums = len(self.fns)
        assert self.nums > 0, f"No images detected in {img_root}."
        if self.nums > 100:
            LOG.warning(f"{self.nums} images detected, the number of recommended images is less than 100.")
        else:
            LOG.info(f"{self.nums} images detected.")
        
        self.batch, self.size = target_size[0], target_size[1:-1]
        if isinstance(mean, list):
            mean = np.array(mean, dtype=np.float32)
        if isinstance(std, list):
            std = np.array(std, dtype=np.float32)
        self.mean, self.std = mean, std

    def _load_img_paths(self, img_root):
        img_paths = []
        for root, _, files in os.walk(img_root):
            for file in files:
                if os.path.splitext(file)[-1].lower() in self.VALID_FORMAT:
                    img_paths.append(os.path.join(root, file))
        return img_paths

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.nums:
            raise StopIteration()
    
        _input = cv2.imread(self.fns[self.index])
        # cv2.imwrite("input.jpg", _input)
        _input = cv2.cvtColor(_input, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("input_bgr.jpg", _input)

        _input = _input.astype(np.float32)
        _input = (_input - 127.5) / 128.0

        if self.mean is not None:
            _input = (_input - self.mean)
        if self.std is not None:
            _input = _input / self.std

        _input = np.expand_dims(_input, axis=0)
        if self.batch > 1:
            _input = np.repeat(_input, self.batch, axis=0).astype(np.float32)
        
        self.index += 1
        return [_input]
