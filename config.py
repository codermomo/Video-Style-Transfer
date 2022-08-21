import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTENT_DATA_ROOT = "data/content/frames"
STYLE_IMG_PATH = "data/style/Vincent_van_Gogh_878.jpg"
NUM_WORKERS = 8
BATCH_SIZE = 1
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_FILENAME = "style_network.pth.tar"
SAVE_IMG_FREQ = 300
STYLIZED_TRAIN_IMG_DIR = "output"
MODEL_DIR = "training_models"


NOISE_COUNT = 20000
NOISE_RANGE = 200 / 255

IMG_SIZE = 512
IMG_CHANNELS = 3
# DOWN_CHANNELS = [32, 64, 128]
DOWN_CHANNELS = [32, 48, 64]
NUM_RES = 5
# UP_CHANNELS = [64, 32]
UP_CHANNELS = [48, 32]
FINAL_CHANNELS = IMG_CHANNELS

VGG19_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]

VGG16_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]

LOSS_NET_CHOICE = "VGG16"

LOSS_NET_CONFIG = {
    "VGG19": {
        "choice": "VGG19",
        "path": "pretrained_models/vgg19-dcbb9e9d.pth",
        "epochs": 1,
        "lr": 1e-3,
        "lambda_content": 17,
        "lambda_style": 25,
        "lambda_regu": 1e-4,
        "lambda_noise": 1e-4,
        "model_cfg": VGG19_CFG,
        "model_map": {
            "ReLU1_2": 3,
            "ReLU2_2": 8,
            "ReLU3_2": 13,
            "ReLU4_2": 22,
        },
        "content_layer": "ReLU4_2",
    },
    "VGG16": {
        "choice": "VGG16",
        "path": "pretrained_models/vgg16_features-amdegroot-88682ab5.pth",
        "epochs": 1,
        "lr": 1e-3 / 4,
        "lambda_content": 6500,
        "lambda_style": 5,
        "lambda_regu": 1e-2,
        "lambda_noise": 1e-2,
        "model_cfg": VGG16_CFG,
        "model_map": {
            "ReLU1_2": 3,
            "ReLU2_2": 8,
            "ReLU3_3": 15,
            "ReLU4_3": 22,
        },
        "content_layer": "ReLU2_2",
    },
}[LOSS_NET_CHOICE]
