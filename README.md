# 🎨 Video Style Transfer
This is an implementation of **single-style-per-model, real-time style transfer** in PyTorch.
![](resource/cutie.gif)

## 📖 Table of Contents
1. [Usage](#💻-usage)
2. [Background](#🗻-background)
3. [Getting Started](#🎈-getting-started)
4. [Acknowledgments](#🎓-acknowledgments)

## 💻 Usage
### Video Stylization
![](resource/train.gif)

### Real-time Video Stylization
Please stay tuned!

### Image Stylization
![](resource/admiralty.jpg)

## 🗻 Background
### Neural Style Transfer
Image/ Video stylization is a technique in the realm of **non-photorealistic rendering (NPR)** which stylizes an input content image/ video with a desired style. With the emerge of deep learning, a new class of algorithm, **neural style transfer (NST)**, has been introduced to tackle the problem by utilizing artificial neural networks. There are mainly two approaches in NST, **optimization-based style transfer** and **feed-forward style transfer**. While the optimization-based approach requires tons of iterations for a single transfer, the feed-forward method makes avail of learned parameters to perform non-linear transformation on content image in only one iteration. The later one, therefore, enables fast generation of stylized contents and makes real-time style transfer possible. It can be further classified into three subcategories: **single style per model, multiple style per model, and arbitrary style per model**. In this project, we place the emphasis on the single-style-per-model style transfer.

### Architecture
Please stay tuned!

### Loss Functions
Please stay tuned!

## 🎈 Getting Started
### Prerequisite
1. Create and activate a virtual environment:
```
python -m venv venv/vst
source venv/vst/Scripts/activate  (for Windows)
source venv/vst/bin/activate  (for Linux)
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Download the pre-trained VGG nets for loss computation if needed: [VGG16](https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth), [VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth).

4. Download some videos for the content dataset from stock video providers such as [Pexels](https://www.pexels.com/videos/), [videvo](https://www.videvo.net/) ... etc if needed. Then, extract the frames using `notebook/data_util.ipynb`.

### Suggested Folder Structure
```
├── data
│   ├── content           # put the raw videos here
│   │   ├── *.mp4
│   │   ├── frames        # put the frames extracted here
│   │   │   ├── *.jpg
│   ├── style             # put the style images here
│   │   ├── *.jpg
│   ├── validation        # put the validation images/ videos here
│   │   ├── *.jpg
│   │   ├── *.mp4
├── network
│   ├── loss_network.py   # the general architecture of pre-trained VGG nets
│   ├── style_network.py  # the architecture of the stylizing network
├── notebook
│   ├── data_util.ipynb   # the utilty for extracting frames from video
├── pretrained_models     # put pretrained VGG nets here
│   ├── *.pth
├── stylization           # the package used by stylize.py
│   ├── content_source    # various data sources and factory are defined here
│   │   ├── *.py
│   ├── __init__.py
│   ├── enum_class.py
│   ├── main.py           # parses arguments and controls data source and processor
│   ├── processor.py      # stylizes input data
├── stylizied_output      # the stylizied output is generated here
│   ├── *.jpg
│   ├── *.mp4
├── train                 # the package used by train.py
│   ├── __init__.py
│   ├── dataset.py        # defines the dataset class for content images
│   ├── loss.py           # defines various loss functions
│   ├── main.py           # parses arguments and controls the training process
├── trained_models        # put the finalized, trained models here, used for stylization
│   ├── *.pth.tar
├── training_models       # the models saved in training are put here
│   ├── *.pth.tar
├── training_output       # the stylized output saved in training are put here for evaluation
│   ├── *.jpg
├── __init__.py
├── config.py             # stores the configuration related to model architecture and loss functions
├── stylize.py            # the entry point for stylization
├── train.py              # the entry point for training
├── utils.py              # contains utility functions used across the entire repository
├── requirements.txt
├── README.md
└── .gitignore
```

### Training
#### Command
```
python train.py \
--device cuda \
--num_workers 8 \
--batch_size 1 \
--img_size 256 \
--content_dir data/content/frames \
--style_path data/style/<style image file> \
--model_name <model name>.pth.tar \
--load_model False \
--save_model True \
--save_freq 300 \
--train_output_dir training_output \
--loss_net_choice VGG16 \
--loss_net_path pretrained_models/vgg16_features-amdegroot-88682ab5.pth \
--epochs 1 \
--lr 2.5e-4 \
--lambda_content 10 \
--lambda_style 100 \
--lambda_regu 1e-2 \
--lambda_noise 1e-5 \
--noise_per_img 20000 \
--noise_range 200
```

#### Parameters
Please stay tuned!

### Stylization
#### Command
```
python stylize.py \
--content_type video \
--source data/validation/<video file> \
--device cuda \
--max_size 1024 \
--model_path trained_models/<trained model name>.pth.tar \
--output_dir stylized_output \
--output_name <name of the stylized output file> \
--real_time True \
--save_output True
```

#### Parameters
Please stay tuned!

## 🎓 Acknowledgments
Please stay tuned!