import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from network.style_network import StylizingNetwork
from network.loss_network import LossNetwork
from .dataset import get_style_image, ContentDataset
import train.loss as loss
import utils
import config as model_cfg


def prepare_config():

    def str2bool(choice):
        true_list = ["yes", "true", "1", "y", "t"]
        false_list = ["no", "false", "0", "n", "f"]
        if choice.lower() in true_list:
            return True
        elif choice.lower() in false_list:
            return False
        raise ValueError(f"Expecting value to be one of {str(true_list + false_list)}, but received {choice}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)

    parser.add_argument("--content_dir", type=str, required=True)
    parser.add_argument("--style_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="training_models")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--load_model", type=str2bool, default=True)
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--save_freq", type=int, default=300)
    parser.add_argument("--train_output_dir", type=str, default="training_output")

    parser.add_argument("--loss_net_choice", type=str, choices=model_cfg.LOSS_NET_CONFIG.keys(), required=True)
    parser.add_argument("--loss_net_path", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--lambda_content", type=float, default=10)
    parser.add_argument("--lambda_style", type=float, default=100)
    parser.add_argument("--lambda_regu", type=float, default=1e-2)
    parser.add_argument("--lambda_noise", type=float, default=1e-5)
    parser.add_argument("--noise_per_img", type=int, default=20000)
    parser.add_argument("--noise_range", type=int, choices=list(range(256)), default=200)
    args = parser.parse_args()

    config = vars(args)
    config["device"] = "cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    config["noise_range"] /= 255
    config["loss_net_config"] = model_cfg.LOSS_NET_CONFIG[config["loss_net_choice"]]
    return config

def train_one_epoch(
    style, loss_network, dataloader, style_optim, scaler, L2, epoch, config
):
    loss_network.eval()
    style_img = get_style_image(config["style_path"], config["img_size"])
    save_image(
        style_img * 0.5 + 0.5,
        os.path.join(config["train_output_dir"], "style_img.png"),
    )
    style_img = style_img.reshape(1, *style_img.shape).to(config["device"])

    constant_noise = utils.get_constant_noise(config["img_size"], config["noise_per_img"], config["noise_range"])
    loop = tqdm(dataloader)

    for idx, frame in enumerate(loop):
        
        if config["device"] == "cuda":
            with torch.cuda.amp.autocast():
                noisy_frame = frame + constant_noise
                noisy_frame.clamp_(min=-1, max=1)
                frame = frame.to(config["device"])
                noisy_frame = noisy_frame.to(config["device"])

                output = style(frame).clamp(min=-1, max=1).to(config["device"])
                noisy_output = (
                    style(noisy_frame).clamp(min=-1, max=1).to(config["device"])
                )
                content_features = loss_network(frame.detach() * 0.5 + 0.5)
                style_features = loss_network(style_img.detach() * 0.5 + 0.5)
                output_features = loss_network(output * 0.5 + 0.5)

                # calculate losses
                content_loss = loss.content_loss(
                    content_features[
                        config["loss_net_config"]["model_map"][
                            config["loss_net_config"]["content_layer"]
                        ]
                    ],
                    output_features[
                        config["loss_net_config"]["model_map"][
                            config["loss_net_config"]["content_layer"]
                        ]
                    ],
                    L2,
                )
                style_loss = torch.stack(
                    [
                        loss.style_loss(
                            style_features[layer],
                            output_features[layer],
                            L2,
                            config["device"],
                        )
                        for layer in config["loss_net_config"]["model_map"].values()
                    ]
                ).sum()
                regularizer = loss.regularizer(output, L2)
                noise_loss = loss.noise_loss(output.detach(), noisy_output, L2)
                total_loss = (
                    config["lambda_content"] * content_loss
                    + config["lambda_style"] * style_loss
                    + config["lambda_regu"] * regularizer
                    + config["lambda_noise"] * noise_loss
                )

            style_optim.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(style_optim)
            scaler.update()
        else:
            noisy_frame = frame + constant_noise
            noisy_frame.clamp_(min=-1, max=1)
            frame = frame.to(config["device"])
            noisy_frame = noisy_frame.to(config["device"])

            output = style(frame).clamp(min=-1, max=1).to(config["device"])
            noisy_output = (
                style(noisy_frame).clamp(min=-1, max=1).to(config["device"])
            )
            content_features = loss_network(frame.detach() * 0.5 + 0.5)
            style_features = loss_network(style_img.detach() * 0.5 + 0.5)
            output_features = loss_network(output * 0.5 + 0.5)

            # calculate losses
            content_loss = loss.content_loss(
                content_features[
                    config["loss_net_config"]["model_map"][
                        config["loss_net_config"]["content_layer"]
                    ]
                ],
                output_features[
                    config["loss_net_config"]["model_map"][
                        config["loss_net_config"]["content_layer"]
                    ]
                ],
                L2,
            )
            style_loss = torch.stack(
                [
                    loss.style_loss(
                        style_features[layer],
                        output_features[layer],
                        L2,
                        config["device"],
                    )
                    for layer in config["loss_net_config"]["model_map"].values()
                ]
            ).sum()
            regularizer = loss.regularizer(output, L2)
            noise_loss = loss.noise_loss(output.detach(), noisy_output, L2)
            total_loss = (
                config["lambda_content"] * content_loss
                + config["lambda_style"] * style_loss
                + config["lambda_regu"] * regularizer
                + config["lambda_noise"] * noise_loss
            )
            style_optim.zero_grad()
            total_loss.backward()
            style_optim.step()

        loop.set_postfix(
            content_loss=float(content_loss),
            regularizer=float(regularizer),
            total_loss=float(total_loss),
            style_loss=float(style_loss),
            noise_loss=float(noise_loss),
        )
        if idx % config["save_freq"] == 0:
            if config["save_model"]:
                utils.save_checkpoint(
                    style,
                    style_optim,
                    os.path.join(
                        config["model_dir"],
                        config["model_name"].split(".")[0]
                        + f"_temp_{idx}"
                        + ".pth.tar",
                    ),
                )
            save_image(
                output * 0.5 + 0.5,
                os.path.join(
                    config["train_output_dir"],
                    f"output_{epoch+1}_{idx+1}.png",
                ),
            )
            save_image(
                noisy_output * 0.5 + 0.5,
                os.path.join(
                    config["train_output_dir"],
                    f"output_{epoch+1}_{idx+1}_noisy.png",
                ),
            )
            save_image(
                frame * 0.5 + 0.5,
                os.path.join(
                    config["train_output_dir"],
                    f"output_{epoch+1}_{idx+1}_origin.png",
                ),
            )
            save_image(
                noisy_frame * 0.5 + 0.5,
                os.path.join(
                    config["train_output_dir"],
                    f"output_{epoch+1}_{idx+1}_noisy_origin.png",
                ),
            )


def train(style, loss_network, dataloader, style_optim, scaler, L2, config):

    for epoch in range(config["epochs"]):

        print(f"epoch {epoch + 1} starts...")
        train_one_epoch(
            style, loss_network, dataloader, style_optim, scaler, L2, epoch, config
        )
        if config["save_model"]:
            utils.save_checkpoint(
                style,
                style_optim,
                os.path.join(config["model_dir"], config["model_name"]),
            )
        print(f"epoch {epoch + 1} ends...")


def main():

    config = prepare_config()

    style = StylizingNetwork().to(config["device"])
    loss_network = LossNetwork(config["loss_net_path"], config["loss_net_config"]).to(config["device"])
    dataset = ContentDataset(config["content_dir"], config["img_size"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    style_optim = optim.Adam(style.parameters(), config["lr"])
    scaler = torch.cuda.amp.GradScaler()
    L2 = nn.MSELoss()

    if not os.path.exists(config["train_output_dir"]):
        os.makedirs(config["train_output_dir"])
    if not os.path.exists(config["model_dir"]):
        os.makedirs(config["model_dir"])

    if config["load_model"]:
        utils.load_checkpoint(
            os.path.join(config["model_dir"], config["model_name"]),
            style,
            style_optim,
            config["lr"],
            config["device"]
        )

    print("start training...")
    train(
        style,
        loss_network,
        dataloader,
        style_optim,
        scaler,
        L2,
        config,
    )
    print("end training...")
