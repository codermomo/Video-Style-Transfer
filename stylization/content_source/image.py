import PIL

import numpy as np

import utils
from .content_source import ContentSource

class Image(ContentSource):
    def __init__(self, infile: str, max_size: str, device: str):
        super().__init__(infile, max_size, device)
    
    def load_data(self):
        img = np.array(PIL.Image.open(self.infile).convert("RGB"))
        yield utils.image2tensor(img, self.max_size, self.device)
    
    def get_fps(self):
        return None