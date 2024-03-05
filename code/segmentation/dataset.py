import torch
import os, cv2
import numpy as np
import utils


# class heartdataset(torch.utils.data.Dataset):
#     def __init__(
#             self,
#             images_dir,
#             masks_dir,
#             class_rgb_values=None,
#             augmentation=None,
#             preprocessing=None,
#     ):
#         self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
#         self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
#         self.class_rgb_values = class_rgb_values
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing

#     def __getitem__(self, i):
#         image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
#         mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
#         mask = utils.one_hot_encode(mask, self.class_rgb_values).astype('float')
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#         if self.preprocessing:
#             sample = self.preprocessing(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
#         return image, mask

#     def __len__(self):
#         return len(self.image_paths)

# def crop_image(image, target_image_dims=[448, 448, 3]):
#     target_size = target_image_dims[0]
#     image_size = len(image)
#     padding = (image_size - target_size) // 2
#     return image[
#         padding:image_size - padding,
#         padding:image_size - padding,
#         :,
#     ]

import cv2
import numpy as np
import albumentations as A


def crop_image(image, mask):
    # Convert mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Find contours of the mask
    contours, _ = cv2.findContours(
        gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get bounding box of the mask
    x, y, w, h = cv2.boundingRect(contours[0])

    # Add some padding to the bounding box
    padding = 10
    x -= padding
    y -= padding
    w += padding * 2
    h += padding * 2

    # Crop image using bounding box
    cropped_image = image[y : y + h, x : x + w]

    return cropped_image


class heartdataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        class_rgb_values=None,
        augmentation=None,
        preprocessing=None,
        crop=False,
        crop_size=320,
        crop_offset=0.25,
    ):
        self.image_paths = [
            os.path.join(images_dir, image_id)
            for image_id in sorted(os.listdir(images_dir))
        ]
        self.mask_paths = [
            os.path.join(masks_dir, image_id)
            for image_id in sorted(os.listdir(masks_dir))
        ]
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.crop = crop
        self.crop_size = crop_size
        self.crop_offset = crop_offset

    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

        if self.crop:
            cropped_image = crop_image(image, mask)
            cropped_mask = crop_image(mask, mask)

            # Resize cropped image and mask to the same size
            resized_crop_size = int(self.crop_size / (1 - 2 * self.crop_offset))
            resized_crop_offset = int(self.crop_offset * resized_crop_size)

            resized_crop_transforms = A.Compose(
                [
                    A.Resize(resized_crop_size, resized_crop_size),
                    A.RandomCrop(self.crop_size, self.crop_size),
                ]
            )

            sample = resized_crop_transforms(image=cropped_image, mask=cropped_mask)

            image, mask = sample["image"], sample["mask"]

        else:
            mask = utils.one_hot_encode(mask, self.class_rgb_values).astype("float")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.image_paths)
