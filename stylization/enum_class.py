from enum import Enum

class ContentSourceChoice(Enum):
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"

    def __str__(self):
        return self.value