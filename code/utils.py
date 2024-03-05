import math
import random

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import splev, splprep
from torch.utils.data import ConcatDataset, DataLoader

import data_loader as dl


def set_seed(seed):
    """
    This function makes the code deterministic using a given seed value.
    It sets the seed for Python, Numpy, PyTorch and CUDA.

    Parameters:
        seed (int): seed value
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class CustomMSELoss(nn.Module):
    """
    Custom Mean Squared Error (MSE) loss with additional penalty for
    the first, last, middle points, and center of mass in each sample.

    This loss function is designed for use with output and label tensors of shape
    (batch_size, num_points, num_coordinates), where each point is represented by
    a pair of coordinates (x, y).

    Args:
        penalty_factor (float, optional): The factor by which the loss for the
            first, last, and middle points will be multiplied. Default is 2.0.
        com_factor (float, optional): The factor by which the loss for the
            center of mass will be multiplied. Default is 0, so that
            the center of mass will not be considered in the loss.
    """

    def __init__(self, loss_penalty_factor=1.0, com_factor=0):
        super(CustomMSELoss, self).__init__()
        self.penalty_factor = loss_penalty_factor
        self.com_factor = com_factor

    def forward(self, outputs, labels):
        assert (
            outputs.shape == labels.shape
        ), f"The shape of outputs and labels should be the same.{outputs.shape} != {labels.shape}"
        batch_size, num_points, num_coordinates = outputs.shape

        # Calculate the squared difference between outputs and labels
        diff = outputs - labels
        squared_diff = diff**2

        # Apply larger penalty to the first, last, and mid points
        first_last_point_diff = squared_diff[:, [0, -1], :].mean() * (
            self.penalty_factor - 1
        )  # subtract 1 to account for the original MSE loss

        mid_point_index = num_points // 2
        if (
            num_points % 2 == 0
        ):  # If there are even number of points, consider two mid-points
            mid_point_diff = squared_diff[
                :, mid_point_index - 1 : mid_point_index + 1, :
            ].mean() * (self.penalty_factor - 1)
        else:  # If there are odd number of points, consider the single mid-point
            mid_point_diff = squared_diff[:, mid_point_index, :].mean() * (
                self.penalty_factor - 1
            )

        # Calculate the MSE for all points
        mse_loss = squared_diff.mean()

        # Calculate the center of mass for outputs and labels separately for x and y coordinates
        com_outputs_x = outputs[:, :, 0].mean(dim=1, keepdim=True)
        com_outputs_y = outputs[:, :, 1].mean(dim=1, keepdim=True)
        com_labels_x = labels[:, :, 0].mean(dim=1, keepdim=True)
        com_labels_y = labels[:, :, 1].mean(dim=1, keepdim=True)

        com_diff_x = (com_outputs_x - com_labels_x) ** 2
        com_diff_y = (com_outputs_y - com_labels_y) ** 2

        com_diff = (com_diff_x + com_diff_y).mean() * self.com_factor

        # Add the extra penalty to the total MSE
        total_loss = mse_loss + first_last_point_diff + mid_point_diff + com_diff

        return total_loss


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


class RotateImageAndPointsTransform:
    """
    This class is used for applying a rotation transformation to both images and corresponding points.
    IMPORTANT: The points should be normalized (i.e., in the range [0, 1]) relative to the image size.

    Attributes:
        angle (float): The angle in degrees to rotate the image and points.

    Methods:
        __call__(sample: dict) -> dict:
            Applies the rotation transformation to an image and its corresponding points.

        rotate_point(x: float, y: float, center_x: float, center_y: float, angle: float) -> Tuple[float, float]:
            Helper method that rotates a point around a center.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image = sample["image"]
        points = sample["landmarks"]

        height, width = image.shape[:2]
        center_x, center_y = width / 2, height / 2

        # Rotate the image
        M = cv2.getRotationMatrix2D((center_x, center_y), self.angle, 1)
        rotated_image = cv2.warpAffine(image, M, (width, height))

        # Update the coordinates
        new_coords = {}
        for key in points:
            # If the points are tensors, convert to numpy
            if isinstance(points[key], torch.Tensor):
                points[key] = points[key].numpy()

            new_coords[key] = []
            for coord in points[key]:
                y_coord, x_coord = coord

                # Scale the coordinates up to image dimensions
                x_scaled = x_coord * width
                y_scaled = y_coord * height

                new_x_scaled, new_y_scaled = self.rotate_point(
                    x_scaled, y_scaled, center_x, center_y, self.angle
                )

                # Scale the coordinates back down
                new_x_coord = new_x_scaled / width
                new_y_coord = new_y_scaled / height

                new_coords[key].append(
                    [new_y_coord, new_x_coord]
                )  # Notice the swap back here

            # Convert back to tensor
            new_coords[key] = torch.tensor(new_coords[key])

        return {"image": rotated_image, "landmarks": new_coords}

    @staticmethod
    def rotate_point(x, y, center_x, center_y, angle):
        angle = math.radians(angle)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        dx = x - center_x
        dy = y - center_y
        new_x = center_x + cos_angle * dx - sin_angle * dy
        new_y = center_y + sin_angle * dx + cos_angle * dy
        return new_x, new_y


class CustomRandomResizedCrop:
    """
    This class applies a custom random resized crop transformation to both the images and the corresponding points
    (landmarks/annotations).

    Importantly, the crop never cuts off any landmarks from the image. The points should be normalized (i.e., in
    the range [0, 1]) relative to the image size.

    Attributes:
        height (int): The height of the output image after cropping and resizing.
        width (int): The width of the output image after cropping and resizing.
        crop_prob (float): The probability of applying the crop. Default is 1.0.

    Methods:
        __call__(sample: dict) -> dict:
            Applies the custom transformation to an image and its corresponding points.

    This class is to be used with the Albumentations library, providing a way to perform the same transformation
    on both an image and its corresponding keypoints. When the image is randomly cropped and resized, the
    coordinates of the keypoints are also updated accordingly.

    The main benefit of this class is that it makes sure the keypoints are still valid after the transformation.
    If a keypoint would be outside the cropped area, the class automatically adjusts the crop so that all
    keypoints are still within the image after the transformation.
    """

    def __init__(self, height, width, crop_prob=0.5):
        self.height = height
        self.width = width
        self.crop_prob = crop_prob

    def __call__(self, sample):
        if random.random() > self.crop_prob:
            return sample

        image = sample["image"]
        points = sample["landmarks"]

        orig_height, orig_width = image.shape[:2]

        # Initialize the crop coordinates to the full image size
        x1, y1, x2, y2 = 0, 0, orig_width, orig_height

        # Calculate bounding box around keypoints to avoid cropping them out
        x_coords = [pt[0] for key in points for pt in points[key]]
        y_coords = [pt[1] for key in points for pt in points[key]]
        keypoint_x1, keypoint_y1, keypoint_x2, keypoint_y2 = (
            min(x_coords),
            min(y_coords),
            max(x_coords),
            max(y_coords),
        )

        # Convert from normalized coordinates to pixel coordinates
        caution_factor = 0.08
        inf_x1 = keypoint_x1 - caution_factor
        inf_x2 = keypoint_x2 + caution_factor
        inf_y1 = keypoint_y1 - caution_factor
        inf_y2 = keypoint_y2 + caution_factor

        keypoint_x1 = max(inf_x1, 0) * orig_width
        keypoint_x2 = min(inf_x2, 1) * orig_width
        keypoint_y1 = max(inf_y1, 0) * orig_height
        keypoint_y2 = min(inf_y2, 1) * orig_height

        # Ensure the crop won't cut off any keypoints
        x1 = np.random.randint(min(keypoint_x1, x1), max(keypoint_x1, x1) + 1)
        y1 = np.random.randint(min(keypoint_y1, y1), max(keypoint_y1, y1) + 1)
        x2 = np.random.randint(min(keypoint_x2, x2), max(keypoint_x2, x2) + 1)
        y2 = np.random.randint(min(keypoint_y2, y2), max(keypoint_y2, y2) + 1)

        # Crop the image

        cropped_image = image[y1:y2, x1:x2]

        # Resize the image back to the desired size
        resized_image = cv2.resize(cropped_image, (self.width, self.height))

        # Adjust the keypoints
        new_coords = {}
        for key in points:
            # If the points are tensors, convert to numpy
            if isinstance(points[key], torch.Tensor):
                points[key] = points[key].numpy()

            new_coords[key] = []
            for coord in points[key]:
                x_coord, y_coord = coord

                # Adjust the coordinates
                new_x_coord = (x_coord * orig_width - x1) / (x2 - x1)
                new_y_coord = (y_coord * orig_height - y1) / (y2 - y1)

                new_coords[key].append([new_x_coord, new_y_coord])

            # Convert back to tensor
            new_coords[key] = torch.tensor(new_coords[key])

        return {"image": resized_image, "landmarks": new_coords}


class CustomTransform:
    """
    This class is used for applying a custom transformation to both images and corresponding points.

    Attributes:
        transforms (list): A list of albumentations transforms to apply to the image and points.
        rotate_angle_range (tuple): A tuple of the form (min_angle, max_angle) that defines the range of angles to rotate the image and points.
        crop_height (int): The height of the output image after cropping and resizing.
        crop_width (int): The width of the output image after cropping and resizing.

    Methods:
        __call__(sample: dict) -> dict:
            Applies the custom transformation to an image and its corresponding points.
    """

    def __init__(
        self, transforms, rotate_angle_range, crop_height, crop_width, crop_prob
    ):
        self.rotate_angle_range = rotate_angle_range
        self.transforms = transforms
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.crop_prob = crop_prob

    def __call__(self, sample):
        rotate_angle = random.uniform(*self.rotate_angle_range)
        sample = RotateImageAndPointsTransform(angle=rotate_angle)(sample)
        sample = CustomRandomResizedCrop(
            self.crop_height, self.crop_width, self.crop_prob
        )(sample)

        for transform in self.transforms:
            sample = transform(**sample)
        return sample


def merge_dataloaders(data_loaders, batch_size=32, shuffle=True, num_workers=8):
    """
    Merge several data loaders into one.

    Parameters:
        data_loaders (list): list of DataLoaders to merge
        batch_size (int, optional): batch size for the merged DataLoader. Defaults to 32.
        shuffle (bool, optional): whether to shuffle the dataset. Defaults to True.
        num_workers (int, optional): number of workers for DataLoader. Defaults to 0.

    Returns:
        DataLoader: merged DataLoader
    """
    merged_dataset = ConcatDataset(
        [data_loader.dataset for data_loader in data_loaders]
    )
    merged_data_loader = DataLoader(
        merged_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return merged_data_loader


def plot_images_two_dataloaders(loaders, num_images=2):
    """
    Plot images with landmarks from two different data loaders.

    Parameters:
        loaders (list): a list of two DataLoader objects
        num_images (int, optional): number of images to plot from each DataLoader. Defaults to 2.
    """
    plt.rcParams["figure.figsize"] = [18, 9]  # Set the figure size

    for i, (loader1_item, loader2_item) in enumerate(zip(loaders[0], loaders[1])):
        images1 = loader1_item["image"]
        landmarks1_lateral = loader1_item["landmarks"]["leaflet_lateral"]
        landmarks1_septal = loader1_item["landmarks"]["leaflet_septal"]

        images2 = loader2_item["image"]
        landmarks2_lateral = loader2_item["landmarks"]["leaflet_lateral"]
        landmarks2_septal = loader2_item["landmarks"]["leaflet_septal"]

        # readjust_keypoints to original size
        landmarks1_lateral = readjust_keypoints(landmarks1_lateral, images1.shape[1:])
        landmarks1_septal = readjust_keypoints(landmarks1_septal, images1.shape[1:])

        landmarks2_lateral = readjust_keypoints(landmarks2_lateral, images2.shape[1:])
        landmarks2_septal = readjust_keypoints(landmarks2_septal, images2.shape[1:])
        # Plot images and landmarks from both loaders
        for (
            image1,
            landmark1_lateral,
            landmark1_septal,
            image2,
            landmark2_lateral,
            landmark2_septal,
        ) in zip(
            images1,
            landmarks1_lateral,
            landmarks1_septal,
            images2,
            landmarks2_lateral,
            landmarks2_septal,
        ):
            # Create a subplot for image and landmarks
            fig, (ax1, ax2) = plt.subplots(1, 2)

            # Set the figure size for each subplot
            ax1.figure.set_size_inches(7, 5)
            ax2.figure.set_size_inches(7, 5)

            # Plot image and landmarks from the first loader
            ax1.imshow(image1, cmap="gray")
            ax1.axis("off")
            ax1.scatter(
                landmark1_lateral[:, 0], landmark1_lateral[:, 1], c="red", s=0.5
            )  # Set smaller point size
            ax1.scatter(
                landmark1_septal[:, 0], landmark1_septal[:, 1], c="blue", s=0.5
            )  # Set smaller point size

            # Plot image and landmarks from the second loader
            ax2.imshow(image2, cmap="gray")
            ax2.axis("off")
            ax2.scatter(
                landmark2_lateral[:, 0], landmark2_lateral[:, 1], c="red", s=0.5
            )  # Set smaller point size
            ax2.scatter(
                landmark2_septal[:, 0], landmark2_septal[:, 1], c="blue", s=0.5
            )  # Set smaller point size

            plt.show()

            # Check if the desired number of images have been plotted
            num_images -= 1
            if num_images == 0:
                return


def load_and_process_data(
    batch_size_train=2,
    batch_size_val_test=16,
    num_workers=8,
    seed=42,
    augment_prop=1,
    normalize_keypts=True,
    rotation=1,
    crop_prob=0.5,
):
    """
    Load and process the data for training, validation and testing.

    The function sets a given seed, loads the data loaders, augments the training data, and merges the data loaders.

    Parameters:
        batch_size_train (int, optional): Batch size for training data. Defaults to 2.
        batch_size_val_test (int, optional): Batch size for validation and test data. Defaults to 16.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 0.
        seed (int, optional): Seed for setting deterministic behavior. Defaults to 42.
        augment_prop (int, optional): Proportion of augmentation for training data. Defaults to 1.
        normalize_keypts (bool, optional): Whether to normalize the output. Defaults to True.
        rotation (int, optional): Rotation angle for augmentation. Defaults to 1.
        crop_prob (float, optional): Probability of applying the crop. Defaults to 0.5.
    Returns:
        dict: A dictionary containing the data loaders for the training, validation, and testing sets.
    """
    set_seed(seed)
    height = 448
    width = 448
    # augment the training data
    base_transform = A.Compose(
        [
            # A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ]
    )

    additional_transform = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomToneCurve(scale=0.1, p=0.5),
            A.GaussNoise(p=0.5),
            A.CLAHE(clip_limit=2, p=0.2),
            base_transform,
        ]
    )

    custom_transform = CustomTransform(
        additional_transform,
        rotate_angle_range=(0 * rotation, 90 * rotation),
        crop_height=height,
        crop_width=width,
        crop_prob=crop_prob,
    )

    base_transform = CustomTransform(
        base_transform,
        rotate_angle_range=(0, 0),
        crop_height=height,
        crop_width=width,
        crop_prob=0,
    )  # this is silly for now but necessary

    # this code above is done in a strange way will fix later.

    # load data loaders
    training_loader = dl.get_data_loader(
        mode="train",
        batch_size=batch_size_train,
        num_workers=num_workers,
        transform=base_transform,
        normalize_keypts=normalize_keypts,
    )

    val_loader = dl.get_data_loader(
        mode="val",
        batch_size=batch_size_val_test,
        num_workers=num_workers,
        transform=base_transform,
        normalize_keypts=normalize_keypts,
    )

    testing_loader = dl.get_data_loader(
        mode="test",
        batch_size=batch_size_val_test,
        num_workers=num_workers,
        transform=base_transform,
        normalize_keypts=normalize_keypts,
    )

    print("> Original training size", len(training_loader.dataset))

    for i in range(augment_prop):

        training_loader_augmented = dl.get_data_loader(
            mode="train",
            batch_size=batch_size_train,
            num_workers=8,
            transform=custom_transform,
            normalize_keypts=normalize_keypts,
        )  # new instance of modified images

        training_loader = merge_dataloaders(
            data_loaders=[training_loader, training_loader_augmented],
            batch_size=batch_size_train,
            num_workers=8,
        )  # merge the two instances of training data

    print("> Original + augmented data size", len(training_loader.dataset))

    dataloaders = {
        x: y
        for x, y in zip(
            ["train", "val", "test"], [training_loader, val_loader, testing_loader]
        )
    }

    return dataloaders


def readjust_keypoints(labels, original_size):
    """
    return the same tensor object but the x coordinates is multiblied by original_size[0] and the y coordinates is multiblied by original_size[1]
    """
    # labels tensor is of shape (batch_size, points, coordinates)
    # if labels have the channel dimension:

    labels[:, :, 0] = (
        labels[:, :, 0] * original_size[1]
    )  # note that is is inverted (y and x)
    labels[:, :, 1] = labels[:, :, 1] * original_size[0]

    return labels
