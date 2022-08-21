import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from style_network import StylizingNetwork
from loss_network import LossNetwork
from dataset import get_style_image, ContentDataset
import loss
import utils
import config
from config import LOSS_NET_CONFIG


def train_one_epoch(
    style, loss_network, dataloader, style_optim, scaler, L2, epoch
):
    loss_network.eval()
    style_img = get_style_image()
    save_image(
        style_img * 0.5 + 0.5,
        os.path.join(config.STYLIZED_TRAIN_IMG_DIR, "style_img.png"),
    )
    style_img = style_img.reshape(1, *style_img.shape).to(config.DEVICE)

    constant_noise = utils.get_constant_noise()
    loop = tqdm(dataloader)

    for idx, frame in enumerate(loop):

        with torch.cuda.amp.autocast():
            noisy_frame = frame + constant_noise
            noisy_frame.clamp_(min=-1, max=1)
            frame = frame.to(config.DEVICE)
            noisy_frame = noisy_frame.to(config.DEVICE)

            output = style(frame).clamp(min=-1, max=1).to(config.DEVICE)
            noisy_output = (
                style(noisy_frame).clamp(min=-1, max=1).to(config.DEVICE)
            )
            content_features = loss_network(frame.detach() * 0.5 + 0.5)
            style_features = loss_network(style_img.detach() * 0.5 + 0.5)
            output_features = loss_network(output * 0.5 + 0.5)

            # calculate losses
            content_loss = loss.content_loss(
                content_features[
                    LOSS_NET_CONFIG["model_map"][
                        LOSS_NET_CONFIG["content_layer"]
                    ]
                ],
                output_features[
                    LOSS_NET_CONFIG["model_map"][
                        LOSS_NET_CONFIG["content_layer"]
                    ]
                ],
                L2,
            )
            style_loss = torch.stack(
                [
                    loss.style_loss(
                        style_features[LOSS_NET_CONFIG["model_map"][layer]],
                        output_features[LOSS_NET_CONFIG["model_map"][layer]],
                        L2,
                    )
                    for layer in LOSS_NET_CONFIG["model_map"].keys()
                ]
            ).sum()
            regularizer = loss.regularizer(output, L2)
            noise_loss = loss.noise_loss(output.detach(), noisy_output, L2)
            total_loss = (
                LOSS_NET_CONFIG["lambda_content"] * content_loss
                + LOSS_NET_CONFIG["lambda_style"] * style_loss
                + LOSS_NET_CONFIG["lambda_regu"] * regularizer
                + LOSS_NET_CONFIG["lambda_noise"] * noise_loss
            )

        style_optim.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(style_optim)
        scaler.update()

        loop.set_postfix(
            content_loss=float(content_loss),
            regularizer=float(regularizer),
            total_loss=float(total_loss),
            style_loss=float(style_loss),
            noise_loss=float(noise_loss),
        )
        if idx % config.SAVE_IMG_FREQ == 0:
            if config.SAVE_MODEL:
                utils.save_checkpoint(
                    style,
                    style_optim,
                    os.path.join(
                        config.MODEL_DIR,
                        config.MODEL_FILENAME.split(".")[0]
                        + f"_temp_{idx}"
                        + ".pth.tar",
                    ),
                )
            save_image(
                output * 0.5 + 0.5,
                os.path.join(
                    config.STYLIZED_TRAIN_IMG_DIR,
                    f"output_{epoch+1}_{idx+1}.png",
                ),
            )
            save_image(
                noisy_output * 0.5 + 0.5,
                os.path.join(
                    config.STYLIZED_TRAIN_IMG_DIR,
                    f"output_{epoch+1}_{idx+1}_noisy.png",
                ),
            )
            save_image(
                frame * 0.5 + 0.5,
                os.path.join(
                    config.STYLIZED_TRAIN_IMG_DIR,
                    f"output_{epoch+1}_{idx+1}_origin.png",
                ),
            )
            save_image(
                noisy_frame * 0.5 + 0.5,
                os.path.join(
                    config.STYLIZED_TRAIN_IMG_DIR,
                    f"output_{epoch+1}_{idx+1}_noisy_origin.png",
                ),
            )


def train(style, loss_network, dataloader, style_optim, scaler, L2, epochs):

    for epoch in range(epochs):

        print(f"epoch {epoch + 1} starts...")
        train_one_epoch(
            style, loss_network, dataloader, style_optim, scaler, L2, epoch
        )
        if config.SAVE_MODEL:
            utils.save_checkpoint(
                style,
                style_optim,
                os.path.join(config.MODEL_DIR, config.MODEL_FILENAME),
            )
        print(f"epoch {epoch + 1} ends...")


if __name__ == "__main__":

    style = StylizingNetwork().to(config.DEVICE)
    loss_network = LossNetwork().to(config.DEVICE)
    dataset = ContentDataset(config.CONTENT_DATA_ROOT)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    style_optim = optim.Adam(style.parameters(), LOSS_NET_CONFIG["lr"])
    scaler = torch.cuda.amp.GradScaler()
    L2 = nn.MSELoss()

    if not os.path.exists(config.STYLIZED_TRAIN_IMG_DIR):
        os.makedirs(config.STYLIZED_TRAIN_IMG_DIR)
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    if config.LOAD_MODEL:
        utils.load_checkpoint(
            os.path.join(config.MODEL_DIR, config.MODEL_FILENAME),
            style,
            style_optim,
            LOSS_NET_CONFIG["lr"],
        )

    print("start training...")
    train(
        style,
        loss_network,
        dataloader,
        style_optim,
        scaler,
        L2,
        LOSS_NET_CONFIG["epochs"],
    )
    print("end training...")
