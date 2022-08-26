from abc import ABC, abstractmethod

class ContentSource(ABC):
    def __init__(self, infile: str, max_size: str, device: str):
        self.infile = infile
        self.max_size = max_size
        self.device = device
    
    @abstractmethod
    def load_data(self):
        pass

    def get_size(self):
        img = next(self.load_data())
        return (img.shape[3], img.shape[2]) # (W, H)

    @abstractmethod
    def get_fps(self):
        pass