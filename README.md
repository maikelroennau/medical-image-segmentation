# Medical Image Segmentation

This repository contains the code and model weights of the papers `A CNN-based approach for joint segmentation and quantification of nuclei and NORs in AgNOR-stained images` and `Automatic Segmentation and Classification of Papanicolaou-stained Cells`.

## Requirements

Windows/Linux
- Anaconda
- CUDA 11.2
- CUDNN 8.2

## Setup

1. Install [Anaconda](https://www.anaconda.com/).
2. Execute the following commands:

```console
git clone https://github.com/maikelroennau/medical-image-segmentation.git
cd medical-image-segmentation
conda env create -n mis --file environment.yml
conda activate mis
```

## Usage

```console
python model_train.py --help

usage: model_train.py [-h] [--encoder ENCODER] [--decoder DECODER] [--dataset DATASET] [--loss LOSS] [--lr LR] [--lr-factor LR_FACTOR] [--classes CLASSES] [--epochs EPOCHS] [--patience PATIENCE] [--batch-size BATCH_SIZE]
                      [--steps STEPS] [--height HEIGHT] [--width WIDTH] [--augmentation {true,false}] [--rgb] [--encoder-freeze] [--save-all] [--gpu GPU] [--name NAME] [--description DESCRIPTION] [--seed SEED] [--resume RESUME]
                      [--resume-epoch RESUME_EPOCH]

Train a segmentation model.

options:
  -h, --help            show this help message and exit
  --encoder ENCODER     The feature extractor of the model.
  --decoder DECODER     The decoder of the model.
  --dataset DATASET     Path to a directory containing 'train', 'validation' and 'test' as sub-directories, each containing an 'images' and 'masks' sub-directories.
  --loss LOSS           Loss function to use during training.
  --lr LR               Learning rate.
  --lr-factor LR_FACTOR
                        Factor to reduce the learning rate when training is stale.
  --classes CLASSES     Number of output dimensions.
  --epochs EPOCHS
  --patience PATIENCE   Number of epochs to wait before reducing the learning rate.
  --batch-size BATCH_SIZE
  --steps STEPS
  --height HEIGHT
  --width WIDTH
  --augmentation {true,false}
                        Enable or disable augmentation in the training set. Defaults to False.
  --rgb                 Whether the input images are RGB.
  --encoder-freeze      Whether or not to freeze the encoder weights.
  --save-all            Whether or no to save all weights. If `True` saves weights after each epoch, if `False`, saves only if better than the previous epoch.
  --gpu GPU             What GPU to use for training. For multi-GPU, pass GPU numbers separated with commas (e.g., `0,1`). For using CPU, pass `-1`.
  --name NAME           Name of the model.
  --description DESCRIPTION
  --seed SEED
  --resume RESUME       Path to the model to be loaded and trained.
  --resume-epoch RESUME_EPOCH
                        The last epoch the model trained.
```

## Examples

Training a model:

```console
python model_train.py --encoder resnet18 --decoder U-Net --dataset path/to/dataset --loss dice --classes 3 --epochs 3 --batch-size 16 --steps 40 --height 1920 --width 2560 --save-all --name MySegmentationModel --description "Segmentation model"
```

Note: The training script expects the dataset directory to contain the following structure:

```console
dataset/
├─ train/
│  ├─ images/
│  ├─ masks/
├─ validation/
│  ├─ images/
│  ├─ masks/
├─ test/
│  ├─ images/
│  ├─ masks/
```

Predicting with a trained model:

```console
python predict.py --model path/to/model.h5 -i path/to/images -o output/path --input-shape 1920x2560x3
```

## Pre-trained models and sample images

### `A CNN-based approach for joint segmentation and quantification of nuclei and NORs in AgNOR-stained images`

You can download pre-trained models (and a few image samples too) for AgNOR slide-image segmentation from here: https://ufrgscpd-my.sharepoint.com/:f:/g/personal/00330519_ufrgs_br/EnzAQbs3_4FHlbxemScpD9IBVKNpGUbXRH0Oqqw7nFkYGA?e=vRbBpS

| **Rank** | **Model file**                              |
|----------|---------------------------------------------|
|     1    | AgNOR_DenseNet-169_LinkNet_1920x2560x3.h5** |
|     2    | AgNOR_DenseNet-169_FPN_1920x2560x3.h5       |
|     3    | AgNOR_ResNet-34_U-Net_1920x2560x3.h5        |

You can download the `AgECOM` dataset used in the paper from [here](https://github.com/maikelroennau/AgECOM).

** This is the preferred model because it obtained the best performance (Dice score) in the [AgECOM](https://github.com/maikelroennau/AgECOM) dataset.


### `Automatic Segmentation and Classification of Papanicolaou-stained Cells`

You can download the pre-trained model (and a few image samples too) for Papanicolaou-stained cell segmentation from here: https://ufrgscpd-my.sharepoint.com/:f:/g/personal/00330519_ufrgs_br/EjGV_28J_PJEgOqitlSWkc4BjpEmGC09RYysHSoYmmPOHg?e=GltEe6

You can download the `UFRGS Pap-OMD` dataset used in the paper from [here](https://github.com/maikelroennau/UFRGS-Pap-OMD).
