import os
import warnings

import numpy as np
import pandas as pd
import segmentation_models_pytorch.utils as smp_utils

import config as config_file
import utils

warnings.filterwarnings("ignore")
import ssl

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import ConcatDataset, DataLoader

from dataset import heartdataset

# disable default certificate verification
if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(
    ssl, "_create_unverified_context", None
):
    ssl._create_default_https_context = ssl._create_unverified_context

# os.environ['WANDB_MODE'] = 'offline' ## for offline mode use this when want to run without syncing data on wandb

# Set seed for reproducibility
utils.seed_everything(config_file.seed)

# Set paths to train, val and test images and masks
DATA_DIR = config_file.path_for_segmentation_data_model
x_train_dir = os.path.join(DATA_DIR, "train/images")
y_train_dir = os.path.join(DATA_DIR, "train/masks")

x_valid_dir = os.path.join(DATA_DIR, "val/images")
y_valid_dir = os.path.join(DATA_DIR, "val/masks")

x_test_dir = os.path.join(DATA_DIR, "test/images")
y_test_dir = os.path.join(DATA_DIR, "test/masks")

class_dict = pd.read_csv(config_file.labels)
# Get class names
class_names = class_dict["name"].tolist()
# Get class RGB values
class_rgb_values = class_dict[["r", "g", "b"]].values.tolist()


# Useful to shortlist specific classes in datasets with large number of classes
select_classes = ["background", "leaflet"]

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

CLASSES = class_names


def build_model(config):

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights,
        classes=len(CLASSES),
        activation=config.activation,
    )
    return model


def train(optimizer, config, train_loader, valid_loader, model, loss, metrics, device):
    # define the training epoch
    train_epoch = smp_utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    # define the validation epoch
    valid_epoch = smp_utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    # Define the learning rate scheduler
    if config.scheduler_mode == "CyclicLR":
        scheduler = CyclicLR(
            optimizer,
            mode="triangular2",
            base_lr=config.base_lr,
            max_lr=config.max_lr,
            step_size_up=len(train_loader) // config.step_size_div_factor,
            cycle_momentum=False,
            gamma=config.lr_scheduler_gamma,
        )
    elif config.scheduler_mode == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=0.0
        )
    elif config.scheduler_mode == "WarmRestart":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(0.1 * config.epochs), T_mult=1, eta_min=0.0
        )

    elif config.scheduler_mode == "StepLR":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=int(0.15 * config.epochs),
            gamma=config.lr_scheduler_gamma,
        )

    # Lists to store learning rate and epoch values
    lr_values, epoch_values = [], []

    # Initialize early stopping parameters
    no_improvement_count = 0

    for i in range(0, config.epochs):
        train_logs = train_epoch.run(train_loader)
        wandb.log({"train_iou_score": train_logs["iou_score"]}, i)
        # wandb.log({"BCEWithLogitsLoss": train_logs["BCEWithLogitsLoss"]}, i)
        wandb.log({"train_dice_loss": train_logs["dice_loss"]}, i)

        valid_logs = valid_epoch.run(valid_loader)
        wandb.log({"val_iou_score": valid_logs["iou_score"]}, i)
        # wandb.log({"BCEWithLogitsLoss": valid_logs["BCEWithLogitsLoss"]}, i)
        wandb.log({"val_dice_loss": valid_logs["dice_loss"]}, i)

        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"current_lr": current_lr}, i)
        lr_values.append(current_lr)
        epoch_values.append(i)
        # Check for improvement in validation IoU score

        if valid_logs["iou_score"] >= best_iou_score:
            best_iou_score = valid_logs["iou_score"]
            no_improvement_count = 0
            best_epoch = i
            # Model saved as best_model.pth
            torch.save(
                model.state_dict(),
                f"segmentation_results/{config.run_name}best_model.pth",
            )
            print("Model saved!")
        else:
            no_improvement_count += 1
            # Check for early stopping condition
            if no_improvement_count >= config.patience:
                print(
                    f"EarlyStopping: No improvement in validation IoU score for the last {no_improvement_count} epochs."
                )
                break

        torch.save(
            model.state_dict(),
            f"segmentation_results/{config.run_name}best_model.pth",
        )


def get_loaders(config, preprocessing_fn):

    # Define the datasets
    train_dataset_original = heartdataset(
        x_train_dir,
        y_train_dir,
        augmentation=utils.get_validation_augmentation(),  ## just makes image multiple of 32
        preprocessing=utils.get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    train_loader = DataLoader(
        train_dataset_original,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
    )
    print("Number of training images: ", len(train_loader.dataset))

    # Function to select the appropriate augmentation function
    if config.augmentation:
        if isinstance(config.augmentation, int):
            augmentation_fns = utils.get_augmentation_by_value([config.augmentation])
        elif isinstance(config.augmentation, list):
            augmentation_fns = utils.get_augmentation_by_value(config.augmentation)
        else:
            raise ValueError(
                "Invalid augmentation value. Must be an integer or a list of integers."
            )

        datasets = []
        for augmentation_fn in augmentation_fns:
            train_dataset_aug = heartdataset(
                x_train_dir,
                y_train_dir,
                augmentation=augmentation_fn,
                preprocessing=utils.get_preprocessing(preprocessing_fn),
                class_rgb_values=select_class_rgb_values,
            )
            datasets.append(train_dataset_aug)

        if datasets:
            combined_dataset = ConcatDataset(datasets)
            train_dataset = ConcatDataset([combined_dataset, train_dataset_original])
        else:
            train_dataset = train_dataset_original

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
        )
        print("Augmentation done.")
        print("Number of training images: ", len(train_loader.dataset))

    valid_dataset = heartdataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=utils.get_validation_augmentation(),
        preprocessing=utils.get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
    )

    return train_loader, valid_loader


def main():
    # Set device: `cuda` or `cpu`
    DEVICE = torch.device(config_file.device)
    wandb.login()
    run = wandb.init(project="new_annot_segmentation", entity="deepvlv", reinit=True)
    run_name = wandb.run.name

    print(run_name)

    config = run.config
    config.run_name = run_name
    config.lr = config_file.lr
    config.epochs = config_file.epochs
    config.batch_size = config_file.batch_size
    config.scheduler_mode = config_file.mode
    config.base_lr = config_file.base_lr
    config.max_lr = config_file.max_lr
    config.encoder = config_file.ENCODER
    config.activation = config_file.ACTIVATION
    config.patience = config_file.patience
    config.scheduler_factor = config_file.scheduler_factor
    config.scheduler_patience = config_file.scheduler_patience
    config.scheduler_threshold = config_file.scheduler_threshold
    config.weight_decay = config_file.weight_decay
    config.IOU_threshold = config_file.threshold
    config.workers = config_file.workers
    config.data_metrics = config_file.data
    config.metrics = config_file.metrics
    config.augmentation = config_file.augmentation
    config.encoder_weights = config_file.ENCODER_WEIGHTS
    config.step_size_div_factor = config_file.step_size_div_factor
    config.lr_scheduler_gamma = config_file.lr_scheduler_gamma

    loss = smp_utils.losses.DiceLoss(
        ignore_channels=[0]
    )  # BCEWithLogitsLoss(ignore_channels=[0])# config_file.loss
    metrics = [smp_utils.metrics.IoU(threshold=config.IOU_threshold)]

    model = build_model(config)

    wandb.watch(model)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config.encoder, config.encoder_weights
    )

    # define optimizer
    optimizer = torch.optim.Adam(
        [
            dict(
                params=model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        ]
    )

    train_loader, valid_loader = get_loaders(config, preprocessing_fn)

    train(
        optimizer,
        config,
        train_loader,
        valid_loader,
        model,
        loss,
        metrics,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
