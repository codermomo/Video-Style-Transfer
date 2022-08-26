import argparse
import os

import torch

from .enum_class import ContentSourceChoice
from .content_source.content_source import ContentSource
from .content_source.content_source_factory import ContentSourceFactory
from .processor import Processor

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_type", type=ContentSourceChoice, choices=list(ContentSourceChoice), required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_size", type=int, default=1024)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="stylized_output")
    parser.add_argument("--output_name", type=str, default="output")
    parser.add_argument("--real_time", type=bool, default=True)
    parser.add_argument("--save_output", type=bool, default=True)
    args = parser.parse_args()
    
    kwargs = {
        "content_type": args.content_type,
        "source": args.source,
        "device": "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu",
        "max_size": args.max_size,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "output_name": args.output_name,
        "real_time": args.real_time,
        "is_video": args.content_type == ContentSourceChoice.VIDEO or args.content_type == ContentSourceChoice.WEBCAM,
        "save_output": args.save_output,
    }

    if not os.path.exists(kwargs["output_dir"]):
        os.mkdir(kwargs["output_dir"])
    
    return kwargs


def main():
    kwargs = prepare_args()

    content_source_factory: ContentSourceFactory = ContentSourceFactory()
    content_source: ContentSource = content_source_factory.get_content_source(kwargs["content_type"], kwargs)

    processor: Processor = Processor(kwargs["model_path"], kwargs["output_dir"], kwargs["output_name"], kwargs["real_time"], kwargs["is_video"], kwargs["save_output"], kwargs["device"])
    processor.stylize(content_source.load_data(), content_source.get_fps(), content_source.get_size())
    