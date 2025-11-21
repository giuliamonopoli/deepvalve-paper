import copy
import os
import ssl
import sys
import time
from tempfile import TemporaryDirectory

sys.path.append("../")
import numpy as np
import segmentation_models_pytorch as segmentation_models
import torch
import torch.nn as nn
import torch.optim as optim
import utils
import wandb
from torch.optim import lr_scheduler


if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(
    ssl,
    "_create_unverified_context",
    None,
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


def build_model(config, device="cpu"):
    """Builds and returns the CNN model based on the provided configuration.

    Parameters:
    - config: A configuration object containing model specifications.
    - device: The device (CPU or GPU) the model should run on.

    Returns:
    - The constructed PyTorch model.

    Raises:
    - ValueError: If the specified model or activation function is not supported.
    """
    if config.model == "unet":
        model = segmentation_models.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            classes=1,  # does not matter for now as we will modify last layer
            activation=None,
        )
    else:
        raise ValueError("Model not supported.")

    # Set activation function
    if config.activation == "relu":
        activation = nn.ReLU()
    elif config.activation == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError("Activation function not supported.")

    # Modify the last layer to match the output size
    model.segmentation_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        nn.Flatten(),
        nn.Linear(16, 10 * 4),
        activation,
        nn.Dropout(p=config.dropout),
        Reshape(),  # Reshape from [batch, 40] to [batch, 10, 4]
    )

    model = model.to(device)
    return model


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

    with TemporaryDirectory() as tmpdirname:
        best_model_path = os.path.join(tmpdirname, "temp_best_model.pt")
        torch.save(model.state_dict(), best_model_path)

        best_loss = np.inf
        no_improvement_count = (
            0  # Initialize the counter of validation loss improvement
        )

        for epoch in range(config.epochs):
            # log current learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]

            if config.use_wandb:
                wandb.log({"current_lr": current_lr}, epoch)
            else:
                print(f"Epoch: {epoch}, Current LR: {current_lr}")

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0  # loss over the batches in epoch

                for i, data in enumerate(dataloaders[phase]):
                    # Get the input and output from the data loader
                    inputs = data["image"]
                    inputs = inputs.unsqueeze(1)  # add a channel dimension
                    inputs = inputs.repeat(
                        1,
                        3,
                        1,
                        1,
                    )  # repeat the channel dimension to create a 3-channel image
                    inputs = inputs.to(device).float()
                    labels_septal = data["landmarks"]["leaflet_septal"]
                    leaflet_lateral = data["landmarks"]["leaflet_lateral"]
                    labels = torch.cat((labels_septal, leaflet_lateral), dim=2).to(
                        device,
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
                if config.use_wandb:
                    wandb.log({f"{phase}_epoch_loss": epoch_loss}, epoch)
                else:
                    print(f"{phase} Loss: {epoch_loss:.4f}")

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
                        f"No improvement in validation loss for {config.patience} epochs. Early stopping training.",
                    )
                    return print_and_return_results(
                        since,
                        model,
                        epoch_loss,
                        best_model_path,
                        best_loss,
                    )

        return print_and_return_results(
            since,
            model,
            epoch_loss,
            best_model_path,
            best_loss,
        )


def prepare_data(config):
    """
    Load and preprocess data according to the configuration.

    Parameters:
    - config: Configuration dictionary containing data loading parameters.

    Returns:
    - dataloaders: A dictionary containing training and validation DataLoader objects.
    """
    dataloaders = utils.load_and_process_data(
        batch_size_train=config.batch_size_train,
        batch_size_val_test=config.batch_size_val_test,
        seed=config.seed,
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
            loss_penalty_factor=config.loss_penalty_factor,
            com_factor=config.com_factor,
        )
    elif config.loss == "huber":
        criterion = nn.HuberLoss()
    elif config.loss == "rmse":
        criterion = utils.RMSELoss()
    else:
        raise ValueError(f"Loss function {config.loss} not implemented")
    return criterion


def configure_scheduler(optimizer, config):
    """
    Configure the learning rate scheduler based on the configuration.

    Parameters:
    - optimizer: The optimization algorithm.
    - config: Configuration dictionary containing scheduler parameters.

    Returns:
    - scheduler: The configured learning rate scheduler.
    """
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
            optimizer,
            T_max=config.epochs,
            eta_min=0.0,
        )
    elif config.schedule == "WarmRestart":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(0.1 * config.epochs),
            T_mult=1,
            eta_min=0.0,
        )
    elif config.schedule == "StepLR":
        scheduler = (
            lr_scheduler.StepLR(
                optimizer,
                step_size=int(0.15 * config.epochs),
                gamma=config.gamma,
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
    - run: The (wandb) run object

    Returns:
    - The updated config with training parameters.
    """
    config = run.config if run else {}
    config.update(
        dict(
            learning_rate=0.0015,
            epochs=1000,
            patience=500,
            batch_size_train=8,
            batch_size_val_test=8,
            gamma=0.48,
            augment_prop=3,
            seed=42,
            model="unet",
            activation="sigmoid",
            schedule="CyclicLR",  # "StepLR" # "CosineAnnealingLR" # "WarmRestart" # "CyclicLR"
            loss="MSE",  # "huber" "rmse"
            rotation=0,  # rotate or not the images in augmentation
            crop_prob=0.8,  # crop or not the images in augmentation
            dropout=0.006,
            normalize_keypts=True,  # whether to normalize the annotations between 0 and 1 using min-max normalization
            loss_penalty_factor=1.016,  # factor to multiply the loss by for the custom loss function. If 1.0, the loss is the same as MSE
            com_factor=0.025,  # factor to multiply the loss by for the center of mass loss. If 0.0, the loss is the same as MSE
            use_wandb=False if run is None else True,
        ),
    )
    return config


def main():
    utils.set_seed(42)
    use_wandb = False  # Set to True to use Weights and Biases
    # Initialize wandb
    if use_wandb:
        run = wandb.init(project="Your_project", entity="Your_entity", reinit=True)
    else:
        run = None

    config = configure_settings(run)
    if use_wandb:
        run_name = wandb.run.name
    else:
        run_name = "Your_run_name"

    run_name += f"{config.loss}"

    print(f"#### Run name: {run_name}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    dataloaders = prepare_data(config)

    # Build the model
    model = build_model(device=device, config=config)

    if use_wandb:
        wandb.watch(model)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = select_criterion(config)

    # Decay LR by a factor of gamma every step_size epochs. Should it be none, set gamma to 0
    config.max_lr_prop = 5 if config.schedule == "CyclicLR" else 0
    config.cycle_step_size = 20 if config.schedule == "CyclicLR" else None
    config.cyclic_mode = "triangular2"  # if config.schedule == "CyclicLR" else None

    scheduler = configure_scheduler(optimizer, config)
    last_model, last_loss, best_model, best_loss = train(
        dataloaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        config,
    )
    if use_wandb:
        run.finish()

    # now save the model to disk
    torch.save(last_model.state_dict(), f"regression_results/{run_name}_last_model.pth")

    torch.save(
        best_model.state_dict(),
        f"unet_regression_results/{run_name}_best_model.pth",
    )


if __name__ == "__main__":
    main()
