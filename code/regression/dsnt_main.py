import copy
import os
import ssl
import sys
import time
from tempfile import TemporaryDirectory

sys.path.append("../")
from code.regression.dsnt_reg import UNetDNST

# from model_dnst import  UNetDNST
import dsntnn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import wandb
from torch.optim import lr_scheduler
from torchvision import models

sys.path.append("../")

if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(
    ssl, "_create_unverified_context", None
):
    ssl._create_default_https_context = ssl._create_unverified_context


class Reshape(nn.Module):
    """A custom PyTorch module that reshapes input tensors to a specified shape."""

    def forward(self, x):
        """Reshapes the input tensor.

        Parameters:
        - x: A tensor to be reshaped.

        Returns:
        - Tensor: Reshaped tensor with the new shape.
        """
        return x.view(x.size(0), -1, 4)


def print_and_return_results(since, model, epoch_loss, best_model_path, best_loss):
    """Prints training results and returns the final and best model along with their losses.

    Parameters:
    - since: Time since the training started.
    - model: The last state of the trained model.
    - epoch_loss: The loss from the last epoch.
    - best_model_path: The path to the best model's state dict.
    - best_loss: The best observed loss during training.

    Returns:
    - Tuple containing the last model, last epoch loss, best model, and best loss.
    """
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val loss: {best_loss:4f}")

    # Load best model weights
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(best_model_path))

    return model, epoch_loss, best_model, best_loss


def dsnt_custom(coords, target_var, heatmaps):
    """
    Computes a custom loss for a set of predicted coordinates against target variables using
    DSNT (Differentiable Spatial to Numerical Transform) operations.

    Parameters:
    - coords (Tensor): The predicted coordinates tensor where each pair represents the x and y coordinates for a point.
    - target_var (Tensor): The target coordinates tensor matching the predicted coordinates' layout.
    - heatmaps (Tensor): The predicted heatmaps tensor of shape representing the spatial distribution of points.

    Returns:
    - loss (Tensor): A scalar tensor representing the combined average loss computed as the sum
                     of Euclidean distance losses and regularization losses across a batch.
    """
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
    loss = dsntnn.average_loss(euc_losses + reg_losses)
    return loss


def train(dataloaders, model, criterion, optimizer, scheduler, device, config):
    """
    Trains a given model with provided data, optimizer, and learning rate scheduler.

    Parameters:
    - dataloaders (dict): A dictionary containing 'train' and 'val' DataLoader objects.
    - model (torch.nn.Module): The neural network model to be trained.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler for learning rate adjustment.
    - device (torch.device): The device to train the model on ('cuda' or 'cpu').
    - config (dict): A configuration dictionary containing training parameters such as epochs and patience.

    Returns:
    - model (torch.nn.Module): The trained model.
    - epoch_loss (float): The loss of the last epoch.
    - best_loss (float): The best validation loss achieved during training.
    """
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

            # log current learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
            wandb.log({"current_lr": current_lr}, epoch)

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

                    inputs = inputs.to(device).float()
                    labels_septal = data["landmarks"]["leaflet_septal"]
                    leaflet_lateral = data["landmarks"]["leaflet_lateral"]
                    labels = torch.cat((labels_septal, leaflet_lateral), dim=1).to(
                        device, dtype=torch.float32
                    )

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
                    return print_and_return_results(
                        since, model, epoch_loss, best_model_path, best_loss
                    )

        return print_and_return_results(
            since, model, epoch_loss, best_model_path, best_loss
        )


def prepare_data(config, seed):
    """
    Load and preprocess data according to the configuration.

    Parameters:
    - config: Configuration dictionary containing data loading parameters.
    - seed: Random seed for reproducibility.

    Returns:
    - dataloaders: A dictionary containing 'train' and 'val' DataLoader objects.
    """
    dataloaders = utils.load_and_process_data(
        batch_size_train=config.batch_size_train,
        batch_size_val_test=config.batch_size_val_test,
        seed=seed,
        augment_prop=config.augment_prop,
        normalize_keypts=config.normalize_keypts,
        rotation=config.rotation,
        crop_prob=config.crop_prob,
    )
    return dataloaders


def select_criterion(config):
    """
    Select the loss function based on the configuration.

    Parameters:
    - config: Configuration dictionary containing loss function parameters.

    Returns:
    - criterion: The selected loss function.
    """

    if config.loss == "MSE":
        criterion = nn.MSELoss()
    elif config.loss == "customMSE":
        criterion = utils.CustomMSELoss(
            loss_penalty_factor=config.loss_penalty_factor, com_factor=config.com_factor
        )
    elif config.loss == "huber":
        criterion = nn.HuberLoss()
    elif config.loss == "rmse":
        criterion = utils.RMSELoss()
    elif config.loss == "dsnt":
        criterion = dsnt_custom
    else:
        raise ValueError(f"Loss function {config.loss} not implemented")
    return criterion


def configure_scheduler(optimizer, config):

    if config.schedule == "CyclicLR":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.learning_rate,
            max_lr=config.max_lr_prop * config.learning_rate,
            step_size_up=config.cycle_step_size,  # epochs to go from base_lr to max_lr
            step_size_down=config.cycle_step_size,  # epochs to go from max_lr to base_lr
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
    else:
        raise ValueError(f"Scheduler {config.schedule} not implemented")

    return scheduler


def configure_settings(run):
    """
    Configure the settings for the model training including learning rate,
    epochs, batch size, and other training parameters.

    Parameters:
    - run: The wandb run object.

    Returns:
    - The updated config with training parameters.
    """
    config = run.config
    config.update(
        dict(
            learning_rate=0.054,
            epochs=2000,
            patience=1000,
            batch_size_train=10,
            batch_size_val_test=10,
            gamma=0.45,
            augment_prop=2,
            schedule=(
                "CyclicLR"  # "StepLR" # "CosineAnnealingLR" # "WarmRestart" # "CyclicLR"
            ),
            loss="huber",  # "customMSE"  # "MSE" # "huber" "rmse"
            rotation=1,  # rotate or not the images in augmentation
            crop_prob=0.75,  # crop or not the images in augmentation
            normalize_keypts=True,  # whether to normalize the annotations between 0 and 1 using min-max normalization
            loss_penalty_factor=1.016,  # factor to multiply the loss by for the custom loss function. If 1.0, the loss is the same as MSE
            com_factor=0.025,  # factor to multiply the loss by for the center of mass loss. If 0.0, the loss is the same as MSE
        )
    )


def main():
    utils.set_seed(42)
    # Initialize wandb
    run = wandb.init(project="dsnt", entity="deepvlv", reinit=True)

    seed = 42
    config = configure_settings(run)
    run_name = wandb.run.name + "_" + config.loss

    print(f"#### Run name: {run_name}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    dataloaders = prepare_data(config, seed)

    # Build the model
    model = UNetDNST().to(device)
    wandb.watch(model)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = select_criterion(config)

    # Decay LR by a factor of gamma every step_size epochs. Should it be none, set gamma to 0
    config.max_lr_prop = (
        5 if config.schedule == "CyclicLR" else None
    )  # *config.learning_rate if config.schedule == "CyclicLR" else None
    config.cycle_step_size = 20 if config.schedule == "CyclicLR" else None
    config.cyclic_mode = "triangular2"  # if config.schedule == "CyclicLR" else None

    scheduler = configure_scheduler(optimizer, config)

    last_model, last_loss, best_model, best_loss = train(
        dataloaders, model, criterion, optimizer, scheduler, device, config
    )

    run.finish()

    # now save the model to disk
    torch.save(best_model.state_dict(), f"dsnt_results/{run_name}_best_model.pth")


if __name__ == "__main__":
    main()
