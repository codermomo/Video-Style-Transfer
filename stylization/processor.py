import os

import cv2
from torchvision.utils import save_image
import keyboard

from network.style_network import StylizingNetwork
import utils


class Processor:
    
    def __init__(self, model_path: str, output_dir: str, output_name: str, show_real_time: bool, is_video: bool, save_output: bool, device: str):
        self.style_network = StylizingNetwork().to(device)
        utils.load_model(model_path, self.style_network, device)
        self.output_dir = output_dir
        self.output_name = output_name
        self.show_real_time = show_real_time
        self.save_output = save_output
        self.is_video = is_video

        self.quit_key = "q"
        self.delay = 1
        self.quit = False
        keyboard.add_hotkey(self.quit_key, self.on_quit_key_pressed)
    
    def stylize(self, contents, fps: int, size: tuple):
        self.style_network.eval()
        video_output = (
            cv2.VideoWriter(os.path.join(self.output_dir, f"{self.output_name}.mp4"), cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
            if self.save_output and self.is_video
            else None
        )

        tensor_output = None
        print(f"Press '{self.quit_key}' to terminate the stylization process ...")
        for idx, content in enumerate(contents):
            tensor_output = self.style_network(content).detach()
            output = utils.tensor2cvimage(tensor_output)

            if self.save_output and self.is_video:
                video_output.write(output)
            
            if self.show_real_time:
                self.show(output)
            
            if cv2.waitKey(self.delay) & 0xFF == ord(self.quit_key) or self.quit:
                print("User interrupted explicitly, ending stylization ...")
                break
        if video_output:
            video_output.release()
        elif self.save_output and not self.is_video:
            self.save_image(tensor_output, os.path.join(self.output_dir, f"{self.output_name}.jpg"))
        cv2.destroyAllWindows()
        print("Done!")

    def on_quit_key_pressed(self):
        self.quit = True
    
    def show(self, output):
        cv2.imshow("Stylized Output", output)

    def save_image(self, tensor_output, outfile: str):
        save_image(
            tensor_output * 0.5 + 0.5,
            outfile,
        )