from ..enum_class import ContentSourceChoice
from .image import Image
from .video import Video
from .webcam import Webcam

class ContentSourceFactory:
    def get_content_source(self, content_type: ContentSourceChoice, kwargs):
        
        infile, max_size, device = kwargs["source"], kwargs["max_size"], kwargs["device"]
        
        if content_type == ContentSourceChoice.IMAGE:
            return Image(infile, max_size, device)
        elif content_type == ContentSourceChoice.VIDEO:
            return Video(infile, max_size, device)
        elif content_type == ContentSourceChoice.WEBCAM:
            return Webcam(infile, max_size, device)
        else:
            raise ValueError(f"content_type is expected to be one of the choices in ContentSourceChoice, but {str(content_type)} received")