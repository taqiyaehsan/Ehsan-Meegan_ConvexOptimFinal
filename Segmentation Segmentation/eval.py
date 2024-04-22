import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from model import UNET
from utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Hyperparameters etc.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_WORKERS = 32
    PIN_MEMORY = True
    TRAIN_IMG_DIR = "data/train_images/"
    TRAIN_MASK_DIR = "data/train_masks/"
    VAL_IMG_DIR = "data/val_images/"
    VAL_MASK_DIR = "data/val_masks/"
    TEST_IMG_DIR = "data/test_images/"
    TEST_MASK_DIR = "data/test_masks/"

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # After we are done training, check testing dataset...
    check_accuracy(test_loader, model, loss_fn, device=DEVICE)

    # print examples to a folder for testing purposes
    save_predictions_as_imgs(
        test_loader, model, folder="saved_images/", device=DEVICE
    )

if __name__ == "__main__":
    main()