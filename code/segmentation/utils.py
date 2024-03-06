import albumentations as album
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import config


def visualize(**images):
    """
    Plot images in one row.

    Parameters:
        **images (dict): Dictionary containing image names as keys and corresponding images as values.
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace("_", " ").title(), fontsize=20)
        plt.imshow(image)
    plt.show()


def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format.

    Parameters:
        label (numpy array): The 2D array segmentation image label.
        label_values (list of tuples): List of color values for each class in the segmentation image.

    Returns:
        numpy array: A 2D array with the same width and height as the input, but with a depth size of num_classes.
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format to a 2D array with only 1 channel.

    Parameters:
        image (numpy array): The one-hot format image.

    Returns:
        numpy array: A 2D array with the same width and height as the input, but with a depth size of 1,
                     where each pixel value is the classified class key.
    """
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    Parameters:
        image (numpy array): Single-channel array where each value represents the class key.
        label_values (list of tuples): List of color values for each class in the segmentation image.

    Returns:
        numpy array: Colour-coded image for segmentation visualization.
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


def get_training_augmentation_1():
    """
    Returns augmentation transforms for training (version 1).

    Returns:
        albumentations.Compose: Augmentation transforms for training (version 1).
    """
    train_transform = [
        album.RandomBrightnessContrast(p=0.3),
        album.HorizontalFlip(p=0.5),
        album.RandomCrop(100, 100, p=1),
        album.PadIfNeeded(
            min_height=448, min_width=448, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(train_transform)


def get_training_augmentation_2():
    """
    Returns augmentation transforms for training (version 2).

    Returns:
        albumentations.Compose: Augmentation transforms for training (version 2).
    """
    train_transform = [
        album.RandomGamma(p=0.3),
        album.VerticalFlip(p=0.7),
        album.PadIfNeeded(
            min_height=448, min_width=448, always_apply=True, border_mode=2
        ),
    ]
    return album.Compose(train_transform)


def get_training_augmentation_3():
    """
    Returns augmentation transforms for training (version 3).

    Returns:
        albumentations.Compose: Augmentation transforms for training (version 3).
    """
    train_transform = [
        album.Rotate(limit=(-90, 90), p=0.9),
        album.CenterCrop(100, 100, p=0.9),
        album.PadIfNeeded(
            min_height=448, min_width=448, always_apply=True, border_mode=4
        ),
    ]
    return album.Compose(train_transform)


def get_training_augmentation_4():
    """
    Returns augmentation transforms for training (version 4).

    Returns:
        albumentations.Compose: Augmentation transforms for training (version 4).
    """
    train_transform = [
        album.PadIfNeeded(
            min_height=448, min_width=448, always_apply=True, border_mode=4
        ),
        album.GaussNoise(var_limit=(0.0, 0.1), p=0.6),
        album.MultiplicativeNoise(multiplier=(0.98, 1.02), p=0.5),
    ]
    return album.Compose(train_transform)


def get_augmentation_by_value(augmentation_values):
    """
    Returns augmentation transforms based on the selected augmentation values.

    Parameters:
        augmentation_values (list): List of integers (1, 2, 3, or 4) representing the selected augmentation versions.

    Returns:
        list of functions: List of augmentation transform functions.
    """
    augmentation_fns = []
    for value in augmentation_values:
        if value == 1:
            augmentation_fns.append(get_training_augmentation_1())
        elif value == 2:
            augmentation_fns.append(get_training_augmentation_2())
        elif value == 3:
            augmentation_fns.append(get_training_augmentation_3())
        elif value == 4:
            augmentation_fns.append(get_training_augmentation_4())
        else:
            raise ValueError(
                f"Invalid augmentation value: {value}. Supported values: 1, 2, or 3."
            )
    return augmentation_fns


def get_validation_augmentation():
    """
    Returns augmentation transforms for validation.

    Returns:
        albumentations.Compose: Augmentation transforms for validation.
    """
    test_transform = [
        album.PadIfNeeded(
            min_height=448, min_width=448, always_apply=True, border_mode=0
        ),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    """
    Convert image data to tensor format.

    Parameters:
        x (numpy array): Image data to be converted to tensor.
        **kwargs: Additional arguments (not used in this function).

    Returns:
        numpy array: Transposed image data in tensor format.
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """
    Construct preprocessing transform for data normalization.

    Parameters:
        preprocessing_fn (callable): Data normalization function (can be specific for each pretrained neural network).

    Returns:
        albumentations.Compose: Augmentation transform for preprocessing.
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
        _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


def seed_everything(seed=42):
    """
    Seeds basic parameters for reproducibility of results.

    Parameters:
        seed (int, optional): Number of the seed. Defaults to 42.
    """
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
