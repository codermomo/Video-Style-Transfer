from .video import Video


class Webcam(Video):
    def __init__(self, infile: str, max_size: str, device: str):
        # throw ValueError if it cannot be casted into int type
        infile = int(infile)
        super().__init__(infile, max_size, device)