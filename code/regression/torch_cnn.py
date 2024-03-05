import albumentations as A
import config as cfg
import data_loader as dl
import torch
import torch.nn as nn
import torch.optim as optim
import utils

import wandb


def load_and_process_data(batch_size_train=89, batch_size_val_test=16, num_workers=0):
    """
    Loads and preprocesses data with augmentation.
    """

    # load data loaders
    training_loader = dl.get_data_loader(
        mode="train", batch_size=batch_size_train, num_workers=num_workers
    )
    val_loader = dl.get_data_loader(
        mode="val", batch_size=batch_size_val_test, num_workers=num_workers
    )
    testing_loader = dl.get_data_loader(
        mode="test", batch_size=batch_size_val_test, num_workers=num_workers
    )

    # augment the training data
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=1,
            ),
            A.CLAHE(p=1),
            A.Equalize(p=1),
            A.RandomToneCurve(scale=0.1, p=1),
        ]
    )

    custom_transform = utils.CustomTransform(transform, rotate_angle_range=(20, 120))

    training_loader_augmented = dl.get_data_loader(
        mode="train",
        batch_size=batch_size_train,
        num_workers=0,
        transform=custom_transform,
    )  # new instance of modified images

    training_loader = utils.merge_dataloaders(
        data_loaders=[training_loader, training_loader_augmented],
        batch_size=len(training_loader.dataset),
        num_workers=0,
    )  # merge the two instances of training data

    # get a single batch from each data loader
    # train_set = next(iter(training_loader))
    # val_set = next(iter(val_loader))
    # test_set = next(iter(testing_loader))

    # extract image data
    # train_data = train_set["image"] / 255.0 # if we want torch tensors, we can use torch.from_numpy()
    # val_data = val_set["image"] / 255.0
    # test_data = test_set["image"] / 255.0

    # extract labels
    # train_labels = train_set["landmarks"]["leaflet_septal"]
    # val_labels = val_set["landmarks"]["leaflet_septal"]
    # test_labels = test_set["landmarks"]["leaflet_septal"]

    return training_loader, val_loader, testing_loader


def build_model(input_shape=(444, 422, 1), device="cpu"):
    """
    Builds and compiles the CNN model.
    """

    # Define the CNN model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
            self.relu1 = nn.LeakyReLU()
            self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv2 = nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            )
            self.relu2 = nn.LeakyReLU()
            self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv3 = nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            )
            self.relu3 = nn.LeakyReLU()
            self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv4 = nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            )
            self.relu4 = nn.LeakyReLU()
            self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

            self.flatten = nn.Flatten()
            expected_size = 27 * 26 * 256

            self.fc1 = nn.Linear(expected_size, 64)
            self.relu5 = nn.ReLU()
            self.fc2 = nn.Linear(
                64, 10 * 2
            )  # 10 * 2 because we have 10 points with x and y coordinates

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.maxpool3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.maxpool4(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu5(x)
            x = self.fc2(x)
            x = x.view(-1, 10, 2)  # Reshape the output to match the expected shape.

            return x

    # Create an instance of the CNN model
    model = CNNModel().to(device)
    return model


def train_one_epoch(epoch_index, model, training_loader, optimizer, criterion, device):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        # Get the input and output from the data loader

        inputs = data["image"].to(device)
        labels = data["landmarks"]["leaflet_septal"].to(device)

        # Set the parameter gradients to zero
        optimizer.zero_grad()
        # Forward pass, backward pass, and optimize
        inputs = inputs.float() / 255.0

        inputs = inputs.unsqueeze(
            1
        )  # add a channel dimension because the model expects it
        outputs = model(inputs)
        loss = criterion(labels.float(), outputs.float())
        loss.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print("[%d, %5d] loss: %.6f" % (epoch_index + 1, i + 1, running_loss / 100))
            wandb.log(
                {"Training Loss": running_loss / 100}
            )  # Log training loss to wandb
            last_loss = running_loss / 100
            running_loss = 0.0

    return last_loss


def train(
    config,
    model,
    training_loader,
    val_loader,
    criterion,
    optimizer,
    device,
):
    for epoch in range(config.epochs):
        print("Epoch", epoch)
        model.train(True)

        train_one_epoch(epoch, model, training_loader, optimizer, criterion, device)
        model.eval()

        running_vloss = 0.0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                vinputs = data["image"].to(device)
                vlabels = data["landmarks"]["leaflet_septal"].to(device)

                vinputs = vinputs.float() / 255.0
                vinputs = vinputs.unsqueeze(1)  # add a channel dimension
                voutputs = model(vinputs)
                vloss = criterion(vlabels.float(), voutputs.float())
                running_vloss += vloss

        avg_vloss = running_vloss / len(val_loader.dataset)
        wandb.log({"Validation Loss": avg_vloss})  # Log validation loss to wandb

        print("Validation loss:", avg_vloss)


def main():
    utils.set_seed(42)
    # Initialize wandb
    run = wandb.init(project="first_pytorch_cnn", entity="deepvlv", reinit=True)
    config = run.config
    config.learning_rate = 0.001
    config.epochs = 20
    config.batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    (training_loader, val_loader, test_loader) = load_and_process_data()

    # Build the model
    model = build_model(device=device)

    wandb.watch(model)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    train(config, model, training_loader, val_loader, criterion, optimizer, device)

    run.finish()

    # now save the model
    torch.save(model.state_dict(), "torchmodel.pth")


if __name__ == "__main__":
    main()
