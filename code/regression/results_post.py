import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.append("../")
from arten_model import UNetDNST
import matplotlib.pyplot as plt

from data_loader import get_data_loader

device = "cpu"
# model = UNetDNST().to(device)
# model.load_state_dict(torch.load("/Users/giuliamonopoli/Desktop/DeepValve Summer Internship/models/newDL_dsnt_best_model.pth",map_location=torch.device('cpu')))
# model.eval()
dataloaders = get_data_loader(
    mode="test", batch_size=4, transform=None, normalize_keypts=True, num_workers=0
)

# dataloaders = utils.load_and_process_data(batch_size_train=8, batch_size_val_test=8, num_workers=0)
# # predict on test set

# ground_truth_septal = []
# ground_truth_lateral = []
# predicted_septal = []
# predicted_lateral = []
# images = []
for i, data in enumerate(dataloaders):
    if i == 1:  # Saving examples from first two batches
        break
    print(data["patient_name"], data["image_id"])
    # inputs = data["image"].to(device).float().unsqueeze(1)
#     outputs = model(inputs)  # shape: (batch_size, 20, 2)

#     labels_septal = data["landmarks"]["leaflet_septal"].to(device)
#     labels_lateral = data["landmarks"]["leaflet_lateral"].to(device)

#     outputs_septal = outputs[:, :10, :]  # shape: (batch_size, 10, 2)
#     outputs_lateral = outputs[:, 10:, :]  # shape: (batch_size, 10, 2)

#     ground_truth_septal.append(labels_septal)
#     ground_truth_lateral.append(labels_lateral)
#     predicted_septal.append(outputs_septal)
#     predicted_lateral.append(outputs_lateral)
#     images.append(inputs)
# torch.save({
#     "images": images,
#     "ground_truth_septal": ground_truth_septal,
#     "ground_truth_lateral": ground_truth_lateral,
#     "predicted_septal": predicted_septal,
#     "predicted_lateral": predicted_lateral
# }, "dsnt_predictions.pth")

saved_data = torch.load("dsnt_predictions.pth")
# import matplotlib.pyplot as plt
print(len(saved_data["images"][0]))
total_images = sum(len(batch) for batch in saved_data["images"])
print(total_images)

# # import matplotlib.pyplot as plt
# # i = 0
# # gt_septal = saved_data['ground_truth_septal'][i].cpu().numpy()
# # pred_septal = saved_data['predicted_septal'][i].cpu().detach().numpy()

# # plt.scatter(
# #         gt_septal[0][:, 0], gt_septal[0][:, 1], s=8, marker=".", c="g", label="true"
# #     )

# # plt.scatter(
# #     pred_septal[0][:, 0], pred_septal[0][:, 1], s=8, marker=".", c="g"
# # )


# # plt.show()
