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
from code.regression.dsnt_reg import UNetDNST
import matplotlib.pyplot as plt

device = "cpu"
model = UNetDNST().to(device)
model.load_state_dict(
    torch.load(
        "/Users/giuliamonopoli/Desktop/DeepValve Summer Internship/models/newDL_dsnt_best_model.pth",
        map_location=torch.device("cpu"),
    )
)
model.eval()

dataloaders = utils.load_and_process_data(
    batch_size_train=8, batch_size_val_test=8, num_workers=0
)
# predict on test set

inputs_lst = []
labels_septal_lst = []
labels_lateral_lst = []
outputs_lst_septal = []
outputs_lst_lateral = []

for i, data in enumerate(dataloaders["test"]):
    if i == 1:
        break
    inputs = data["image"]
    inputs = inputs.to(device).float()
    inputs = inputs.unsqueeze(1)
    outputs = model(inputs)  # shape: (batch_size, 10, 4)
    print("outputs.shape", outputs.shape)

    original_size = inputs.shape[-2:]

    inputs_lst.append(inputs)

    # inputs = F.interpolate(inputs, size=(224, 224))
    labels_septal = data["landmarks"]["leaflet_septal"]
    labels_septal = utils.readjust_keypoints(labels_septal, original_size)
    labels_septal_lst.append(labels_septal)

    labels_lateral = data["landmarks"]["leaflet_lateral"]
    labels_lateral = utils.readjust_keypoints(labels_lateral, original_size)
    labels_lateral_lst.append(labels_lateral)

    outputs_septal = outputs[:, :10, :]  # shape: (batch_size, 10, 2)
    outputs_lateral = outputs[:, 10:, :]  # shape: (batch_size, 10, 2)
    # detach

    outputs_septal = outputs_septal.cpu().detach().numpy()
    outputs_lateral = outputs_lateral.cpu().detach().numpy()

    outputs_septal = utils.readjust_keypoints(outputs_septal, original_size)
    outputs_lateral = utils.readjust_keypoints(outputs_lateral, original_size)

    outputs_lst_septal.append(outputs_septal)
    outputs_lst_lateral.append(outputs_lateral)

# now plot the images with the predicted keypoints

fig = plt.figure(figsize=(10, 10))
for j in range(1, 8):
    ax = plt.subplot(4, 4, j)
    plt.imshow(inputs_lst[0][j][0], cmap="gray")

    plt.scatter(
        labels_septal_lst[0][j][:, 0],
        labels_septal_lst[0][j][:, 1],
        s=8,
        marker=".",
        c="g",
        label="true",
    )

    plt.scatter(
        labels_lateral_lst[0][j][:, 0],
        labels_lateral_lst[0][j][:, 1],
        s=8,
        marker=".",
        c="g",
    )

    plt.scatter(
        outputs_lst_septal[0][j][:, 0],
        outputs_lst_septal[0][j][:, 1],
        s=8,
        marker=".",
        c="r",
    )

    plt.scatter(
        outputs_lst_lateral[0][j][:, 0],
        outputs_lst_lateral[0][j][:, 1],
        s=8,
        marker=".",
        c="r",
        label="predicted",
    )

    # add legend
    plt.legend(loc="upper right", prop={"size": 6})
    # make font smaller
    plt.tick_params(axis="both", which="major", labelsize=1)


plt.show()
