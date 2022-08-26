from .video import Video


class Webcam(Video):
    def __init__(self, infile: str, max_size: str, device: str):
        super().__init__(infile, max_size, device)
        # throw ValueError if it cannot be casted into int type
        self.infile = int(self.infile)