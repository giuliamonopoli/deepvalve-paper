import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.interpolate import splev, splprep
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'code', 'segmentation'))
import config


def get_spline_pts(leaf_pts, num_pts=10):
    """
    Generate spline points based on the given leaf points.

    Parameters:
        leaf_pts (array_like): The input leaf points.
        num_pts (int, optional): The number of points to generate on the spline. Default is 10.

    Returns:
        array_like: The spline points.

    Notes:
        - The function uses `splprep` to obtain a B-spline representation of the curve that passes through the given points.
        - The B-spline representation consists of three elements: knots, coefficients, and the degree of the spline.
        - The knots define the curve, the coefficients are the weights of the curve, and the degree is the degree of the polynomial that defines the curve.
        - The function uses `splev` to evaluate the B-spline representation and obtain the points on the curve.

    """
    tck, u = splprep(leaf_pts.T, s=0, k=min(3, len(leaf_pts) - 1))
    x = np.linspace(u.min(), u.max(), num_pts)
    y = np.array(splev(x, tck, der=0))
    return y.T


## to create a single leaflet
# Define a function to create a mask from an image and a dictionary
def create_mask_polygon(image, dictionary, no_of_points=40):
    # Extract the 'leaflet_septal' key values and convert them to a list of tuples
    coordinates = dictionary["leaflet_septal"]
    coordinates = get_spline_pts(coordinates, no_of_points)
    coordinates = [tuple(p) for p in coordinates]

    # Create a blank image with the same shape as the input image
    mask = Image.new("L", (image.shape[1], image.shape[0]), 0)

    # Draw a polygon on the blank image using the coordinates
    ImageDraw.Draw(mask).polygon(coordinates, outline=1, fill=1)

    # Convert the blank image to a numpy array
    mask = np.array(mask)

    # # Optionally, display the images and the masks
    # for i in range(len(x)):
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(x[i], cmap='gray')
    # plt.title('Image')
    # plt.subplot(1,2,2)
    # plt.imshow(masks[i], cmap='gray')
    # plt.title('Mask')
    # plt.show()

    # Return the mask as a binary array
    return mask.astype(bool)


def create_mask_line(image, annotation, no_of_points=40, thickness=2):
    """
    Plot points on an image using a binary mask with added thickness for both leaflets

    Parameters:
        image (ndarray): NumPy array containing the image.
        leaflet_septal (array_like): NumPy array containing the septal leaflet coordinates.
        leaflet_lateral (array_like): NumPy array containing the lateral leaflet coordinates.
        no_of_points (int): Number of points to generate on the B-spline (default is 40).
        thickness (int): Thickness of the plotted points (default is 2).

    Returns:
        ndarray: The final mask containing the combined points from both leaflet sets.
    """

    coordinates_septal = get_spline_pts(annotation.get("leaflet_septal"), no_of_points)
    coordinates_lateral = get_spline_pts(
        annotation.get("leaflet_lateral"), no_of_points
    )

    # Convert floating-point coordinates to integers
    coordinates_array_septal = coordinates_septal.astype(int)
    coordinates_array_lateral = coordinates_lateral.astype(int)

    # Create blank masks with the same size as the image
    mask_septal = np.zeros_like(image)
    mask_lateral = np.zeros_like(image)

    # Plot the points on the masks with added thickness
    for x, y in coordinates_array_septal:
        mask_septal[
            max(0, y - thickness) : min(mask_septal.shape[0], y + thickness + 1),
            max(0, x - thickness) : min(mask_septal.shape[1], x + thickness + 1),
        ] = 1

    for x, y in coordinates_array_lateral:
        mask_lateral[
            max(0, y - thickness) : min(mask_lateral.shape[0], y + thickness + 1),
            max(0, x - thickness) : min(mask_lateral.shape[1], x + thickness + 1),
        ] = 1

    # Combine the masks (OR operation) to get the final mask
    final_mask = np.logical_or(mask_septal, mask_lateral)

    return final_mask.astype(bool)


# Define a custom dataset class
class ImageMaskDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        # Store the images and masks as attributes
        self.images = images
        self.masks = masks
        # Store the transform as an attribute
        self.transform = transform

    def __len__(self):
        # Return the length of the list of images or masks
        return len(self.images)

    def __getitem__(self, idx):
        # Get an image and a mask at a given index
        image = self.images[idx]
        mask = self.masks[idx]
        # Convert them to PIL Image objects
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        # Apply the transform if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        # Return a tuple of image and mask
        return image, mask


def create_dataloaders_with_masks_polygons(
    training_loader, val_loader, testing_loader, no_of_points
):
    """Creates dataloaders with masks for training, validation, and testing.

    Args:
      training_loader: The training dataloader.
      val_loader: The validation dataloader.
      testing_loader: The testing dataloader.

    Returns:
      A tuple of dataloaders with masks.
    """
    masks_training = []
    masks_val = []
    masks_testing = []

    for image, dictionary in zip(
        training_loader.dataset.images, training_loader.dataset.landmarks
    ):
        mask = create_mask_polygon(image, dictionary, no_of_points)
        masks_training.append(mask)

    for image, dictionary in zip(
        val_loader.dataset.images, val_loader.dataset.landmarks
    ):
        mask = create_mask_polygon(image, dictionary, no_of_points)
        masks_val.append(mask)

    for image, dictionary in zip(
        testing_loader.dataset.images, testing_loader.dataset.landmarks
    ):
        mask = create_mask_polygon(image, dictionary, no_of_points)
        masks_testing.append(mask)

    train = ImageMaskDataset(training_loader.dataset.images, masks_training)
    val = ImageMaskDataset(val_loader.dataset.images, masks_val)
    testing = ImageMaskDataset(testing_loader.dataset.images, masks_testing)

    return (train, val, testing)


def create_dataloaders_with_masks_line(
    training_loader, val_loader, testing_loader, no_of_points=10, mask_thickness=2
):
    """Creates dataloaders with masks for training, validation, and testing.

    Args:
      training_loader: The training dataloader.
      val_loader: The validation dataloader.
      testing_loader: The testing dataloader.

    Returns:
      A tuple of dataloaders with masks.
    """
    masks_training = []
    masks_val = []
    masks_testing = []

    for image, dictionary in zip(
        training_loader.dataset.images, training_loader.dataset.landmarks
    ):
        mask = create_mask_line(image, dictionary, no_of_points, mask_thickness)
        masks_training.append(mask)

    for image, dictionary in zip(
        val_loader.dataset.images, val_loader.dataset.landmarks
    ):
        mask = create_mask_line(image, dictionary, no_of_points, mask_thickness)
        masks_val.append(mask)

    for image, dictionary in zip(
        testing_loader.dataset.images, testing_loader.dataset.landmarks
    ):
        mask = create_mask_line(image, dictionary, no_of_points, mask_thickness)
        masks_testing.append(mask)

    train = ImageMaskDataset(training_loader.dataset.images, masks_training)
    val = ImageMaskDataset(val_loader.dataset.images, masks_val)
    testing = ImageMaskDataset(testing_loader.dataset.images, masks_testing)

    return (train, val, testing)


def save_images_masks(dataset, data_type, mask_type, folder_path):
    """
    dataset: an instance of the ImageMaskDataset class
    data_type: a string representing the type of data, e.g. 'train', 'test', or 'val'
    mask_type: line or polygon
    folder_path: location to save the segmentation files
    """
    mask_subfolder = (
        "segmentation_lines" if mask_type == "line" else "segmentation_polygons"
    )
    no_of_points = config.no_of_points

    if mask_type == "line":
        mask_thickness = config.mask_thickness
        folder_path = f"{folder_path}/{mask_subfolder}_{no_of_points}_{mask_thickness}/"
    else:
        folder_path = f"{folder_path}/{mask_subfolder}_{no_of_points}/"

    # Create the directories
    os.makedirs(f"{folder_path}{data_type}/images", exist_ok=True)
    os.makedirs(f"{folder_path}{data_type}/masks", exist_ok=True)

    # Save the images and masks
    for i in range(len(dataset)):
        image, mask = dataset.images[i], dataset.masks[i]
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image.save(f"{folder_path}{data_type}/images/image_{i}.png")
        mask.save(f"{folder_path}{data_type}/masks/image_{i}.png")


def save_images_and_masks_in_folders(
    train_loader, val_loader, test_loader, mask_type, base_folder_path="../"
):
    # Save images and masks for train
    save_images_masks(train_loader, "train", mask_type, base_folder_path)

    # Save images and masks for val
    save_images_masks(val_loader, "val", mask_type, base_folder_path)

    # Save images and masks for test
    save_images_masks(test_loader, "test", mask_type, base_folder_path)
