import cv2

import utils
from .content_source import ContentSource


class Video(ContentSource):
    def __init__(self, infile: str, max_size: str, device: str):
        super().__init__(infile, max_size, device)
        self.cap = cv2.VideoCapture(self.infile)
    
    def load_data(self):
        if self.cap is None or not self.cap.isOpened():
            raise Exception(f"Error opening the video stream or file: {self.infile}")
        
        while True:
            success, frame = self.cap.read()
            if not success:
                return
            yield utils.cvimage2tensor(frame, self.max_size, self.device)
    
    def get_fps(self):
        return int(self.cap.get(cv2.CAP_PROP_FPS))