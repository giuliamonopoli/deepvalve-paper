import copy
import os
import ssl
import sys
import time
from tempfile import TemporaryDirectory
import sys

sys.path.append("../")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import utils
from torch.optim import lr_scheduler
from torchvision import models

import wandb

sys.path.append("../")

if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(
    ssl, "_create_unverified_context", None
):
    ssl._create_default_https_context = ssl._create_unverified_context


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1, 4)


def print_and_return_results(since, model, epoch_loss, best_model_path, best_loss):
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:4f}")

    # load best model weights
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(best_model_path)),  # load the best model

    return (
        model,  # model is last model
        epoch_loss,  # epoch_loss is the last loss
        best_model,
        best_loss,
    )


def build_model(config, device="cpu"):
    """
    model=config.model, dropout=config.dropout, activation=config.activation)
    Builds and compiles the CNN model.
    """

    if config.model == "resnet50":
        model = models.resnet50(weights="DEFAULT")
    else:
        raise ValueError("Model not supported.")

    if config.activation == "relu":
        activation = nn.ReLU()
    elif config.activation == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError("Activation function not supported.")
    # Modify the last layer to match the output size
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 10 * 4),
        activation,  # Adding ctivation function
        nn.Dropout(p=config.dropout),  # Adding dropout layer
        Reshape(),  # Reshape from [batch, 40] to [batch, 10, 4]
    )

    # Move model to the device
    model = model.to(device)

    return model


def train(dataloaders, model, criterion, optimizer, scheduler, device, config):
    since = time.time()
    # create a temporary directory to store training checkpoints
    with TemporaryDirectory() as tmpdirname:
        best_model_path = os.path.join(tmpdirname, "temp_best_model.pt")

        torch.save(model.state_dict(), best_model_path)
        best_loss = np.inf
        no_improvement_count = (
            0  # Initialize the counter for tracking validation loss improvement
        )

        for epoch in range(config.epochs):
            # print(f"Epoch {epoch}/{config.epochs - 1}")
            # print("-" * 10)

            # log current learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
            wandb.log({"current_lr": current_lr}, epoch)

            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0  # loss over the batches in epoch

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase]):
                    # Get the input and output from the data loader
                    inputs = data["image"]
                    inputs = inputs.unsqueeze(1)  # add a channel dimension
                    inputs = inputs.repeat(
                        1, 3, 1, 1
                    )  # repeat the channel dimension to create a 3-channel image
                    inputs = inputs.to(device).float()
                    labels_septal = data["landmarks"]["leaflet_septal"]
                    leaflet_lateral = data["landmarks"]["leaflet_lateral"]
                    labels = torch.cat((labels_septal, leaflet_lateral), dim=2).to(
                        device
                    )

                    # Set the parameter gradients to zero
                    optimizer.zero_grad()

                    # Forward pass
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        loss = criterion(outputs.float(), labels.float())

                        # Backward pass + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()

                if phase == "train" and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                wandb.log({f"{phase}_epoch_loss": epoch_loss}, epoch)

                # print(f"{phase} Loss: {epoch_loss:.4f}")

                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_path)
                    no_improvement_count = (
                        0  # Reset the counter when we have a new best loss
                    )
                elif phase == "val" and epoch_loss >= best_loss:
                    no_improvement_count += (
                        1  # Increment the counter when there's no improvement in loss
                    )

                # Early stopping
                if no_improvement_count >= config.patience:
                    print(
                        f"No improvement in validation loss for {config.patience} epochs. Early stopping training."
                    )
                    return print_and_return_results(
                        since, model, epoch_loss, best_model_path, best_loss
                    )

            # print()  # add a new line

        return print_and_return_results(
            since, model, epoch_loss, best_model_path, best_loss
        )


def main():
    utils.set_seed(42)
    # Initialize wandb
    run = wandb.init(
        project="new_annot_resnet_reg_exploit", entity="deepvlv", reinit=True
    )

    # get wandb run name

    config = run.config
    config.learning_rate = 0.058
    config.epochs = 1000
    config.patience = 500
    config.batch_size_train = 16
    config.batch_size_val_test = 16
    config.gamma = 0.4  # 0.6
    config.augment_prop = 1
    config.seed = 42
    config.model = "resnet50"
    config.activation = "sigmoid"
    config.schedule = (
        "CyclicLR"  # "StepLR" # "CosineAnnealingLR" # "WarmRestart" # "CyclicLR"
    )
    config.loss = "huber"  # "MSE" # "huber" "rmse"
    run_name = wandb.run.name + "_" + config.loss

    print(f"#### Run name: {run_name}")
    config.rotation = 1  # rotate or not the images in augmentation
    config.crop_prob = 0.327  # crop or not the images in augmentation
    config.dropout = 0.0257
    config.normalize_keypts = True  # whether to normalize the annotations between 0 and 1 using min-max normalization
    config.loss_penalty_factor = 1.016  # factor to multiply the loss by for the custom loss function. If 1.0, the loss is the same as MSE
    # because of this we will not add the MSE loss, only the custom loss
    config.com_factor = 0.025  # factor to multiply the loss by for the center of mass loss. If 0.0, the loss is the same as MSE
    # make dataframe with run name and configs
    df = pd.DataFrame.from_dict({k: [v] for k, v in config.items()})

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    dataloaders = utils.load_and_process_data(
        batch_size_train=config.batch_size_train,
        batch_size_val_test=config.batch_size_val_test,
        seed=config.seed,
        augment_prop=config.augment_prop,
        normalize_keypts=config.normalize_keypts,
        rotation=config.rotation,
        crop_prob=config.crop_prob,
    )

    # Build the model
    model = build_model(device=device, config=config)
    wandb.watch(model)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    if (
        config.loss == "MSE"
    ):  # this is not used anymore because we can use the custom loss function with penalty factor 1 (same as MSE)
        criterion = nn.MSELoss()
    elif config.loss == "customMSE":
        criterion = utils.CustomMSELoss(
            loss_penalty_factor=config.loss_penalty_factor, com_factor=config.com_factor
        )
    elif config.loss == "huber":
        criterion = nn.HuberLoss()
    elif config.loss == "rmse":
        criterion = utils.RMSELoss()

    # else:
    #    raise ValueError("Loss function not implemented")

    # Decay LR by a factor of gamma every step_size epochs. Should it be none, set gamma to 0
    config.max_lr_prop = (
        5 if config.schedule == "CyclicLR" else 0
    )  # *config.learning_rate if config.schedule == "CyclicLR" else None
    config.cycle_step_size = 20 if config.schedule == "CyclicLR" else None
    config.cyclic_mode = "triangular2"  # if config.schedule == "CyclicLR" else None

    if config.schedule == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.learning_rate,
            max_lr=config.max_lr_prop * config.learning_rate,
            step_size_up=config.cycle_step_size,  # 5 epochs to go from base_lr to max_lr
            step_size_down=config.cycle_step_size,  # 5 epochs to go from max_lr to base_lr
            mode=config.cyclic_mode,
            gamma=config.gamma,
            cycle_momentum=False,
        )
    elif config.schedule == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=0.0
        )
    elif config.schedule == "WarmRestart":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(0.1 * config.epochs), T_mult=1, eta_min=0.0
        )

    elif config.schedule == "StepLR":
        scheduler = (
            lr_scheduler.StepLR(
                optimizer, step_size=int(0.15 * config.epochs), gamma=config.gamma
            )
            if config.gamma
            else None
        )

    last_model, last_loss, best_model, best_loss = train(
        dataloaders, model, criterion, optimizer, scheduler, device, config
    )
    df["best_loss"] = [best_loss]
    df["last_loss"] = [last_loss]

    # df.to_csv("regression_results_exploitation/metadata.csv", index=False, mode="a")
    run.finish()
    # now save the model to disk
    # torch.save(last_model.state_dict(), f"regression_results/{run_name}_last_model.pth")
    torch.save(
        best_model.state_dict(),
        f"resnet_regression_results_exploitation/{run_name}_best_model.pth",
    )


if __name__ == "__main__":
    main()
