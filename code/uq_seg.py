
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import segmentation_models_pytorch as smp
import seaborn as sns
import cv2
import dataset
import utils
import config
from  segmentation_models_pytorch.utils.metrics import Fscore
# Paths
x_test_dir = "/home/giulia/deepvalve/data/segmentation_data/segmentation_lines_30_2/test/images/"
y_test_dir = "/home/giulia/deepvalve/data/segmentation_data/segmentation_lines_30_2/test/masks/"


# Load class dictionary
class_dict = pd.read_csv(config.labels)
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()
select_classes = ['background', 'leaflet']
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

# Load model
ENCODER = config.ENCODER
ENCODER_WEIGHTS = config.ENCODER_WEIGHTS
CLASSES = class_names
ACTIVATION = config.ACTIVATION  
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model = model.to("cpu")
model_path = "path/to/your/model.pth"  # Replace with your model path
model.load_state_dict(torch.load(model_path))

test_dataset = dataset.heartdataset(
    x_test_dir,
    y_test_dir,
    augmentation=utils.get_validation_augmentation(),
    preprocessing=utils.get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

def enable_dropout(model, dropout_prob):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            new_layer = nn.Sequential(
                child,
                nn.Dropout2d(p=dropout_prob)
            )
            setattr(model, name, new_layer)
        else: 
            enable_dropout(child, dropout_prob)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def compute_confidence_intervals(data, confidence=0.95):
    n = len(data)
    mean_data = np.mean(data)
    std_data = np.std(data)
    z_value = norm.ppf(1 - (1 - confidence) / 2)
    sem = std_data / np.sqrt(n)  
    margin_of_error = z_value * sem
    lower_bound = mean_data - margin_of_error
    upper_bound = mean_data + margin_of_error
    return mean_data, lower_bound, upper_bound


def dice_coefficient(y_true, y_pred, class_index):
    intersection = np.sum((y_true == class_index) & (y_pred == class_index))
    union = np.sum(y_true == class_index) + np.sum(y_pred == class_index)
    return (2. * intersection) / (union + 1e-8)
f = Fscore()
# Perform MC inference
forward_passes = 50
dropout_predictions = []
dropout_dice_scores = []
drop_prob = 0.2
for i in range(forward_passes):
    print(f"Starting forward pass {i + 1}/{forward_passes}")
    model.eval()
    enable_dropout(model.decoder, drop_prob)
   
    predictions = []
    dice_scores = []
    for idx in range(len(test_dataset)):
        images, gt_mask = test_dataset[idx]
        gt_mask = np.transpose(gt_mask, (1, 2, 0))
        x_tensor = torch.from_numpy(images).to("cpu").unsqueeze(0)  

        with torch.no_grad():
            output = model(x_tensor).detach().cpu().squeeze() # Shape: (1, 2, 448, 448)
            predictions.append(output.numpy() )  # Append to predictions list
            output = np.transpose(output, (1, 2, 0))
            output =utils.reverse_one_hot(output)
            gt_mask = utils.reverse_one_hot(gt_mask)
            dice_scores.append(f(torch.from_numpy(gt_mask),output))
            
    predictions = np.array(predictions)  # Shape: (n_samples, 2, 448, 448)
    dice_scores = np.array(dice_scores)
    dropout_predictions.append(predictions)
    dropout_dice_scores.append(dice_scores)

dropout_predictions = np.array(dropout_predictions)  # Shape: (forward_passes, n_samples, 2, 448, 448)
dropout_dice_scores = np.array(dropout_dice_scores)  # Shape: (forward_passes, n_samples)
print("dropout_dice_scores shape is ",dropout_dice_scores.shape)

sample_dice_scores = np.median(dropout_dice_scores, axis=0)
print("sample_dice_scores shape is ",sample_dice_scores.shape)

sample_conf_intervals = [compute_confidence_intervals(dropout_dice_scores[:, i], confidence=0.95) for i in range(dropout_dice_scores.shape[1])]

# Calculate Dice scores for all cases across all runs with confidence intervals
overall_dice_scores = dropout_dice_scores.flatten()
print(f"Overall dice score flatten shape is {overall_dice_scores}")
overall_mean_dice, overall_lower_bound, overall_upper_bound = compute_confidence_intervals(overall_dice_scores, confidence=0.95)

# Print results
print("Sample Dice Scores with Confidence Intervals:")
for i, (mean_dice, lower_bound, upper_bound) in enumerate(sample_conf_intervals):
    print(f"Sample {i + 1}: Dice Score = {mean_dice:.4f}, 95% CI = [{lower_bound:.4f}, {upper_bound:.4f}]")

print(f"\nOverall Dice Score: {overall_mean_dice:.4f}, 95% CI = [{overall_lower_bound:.4f}, {overall_upper_bound:.4f}]")

# Save results to CSV
results = {
    "Sample": list(range(1, len(sample_dice_scores) + 1)),
    "Mean Dice Score": sample_dice_scores,
    "Lower Bound CI": [ci[1] for ci in sample_conf_intervals],
    "Upper Bound CI": [ci[2] for ci in sample_conf_intervals]
}
results_df = pd.DataFrame(results)
output_csv_path = f"/home/giulia/deepvalve/results/uncertainty_analysis_{drop_prob}b.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
global_results = {
    "Overall Mean Dice Score": [overall_mean_dice],
    "Overall Lower Bound CI": [overall_lower_bound],
    "Overall Upper Bound CI": [overall_upper_bound]
}
global_output_csv_path = f"path/to/global_results_{drop_prob}b.csv"

global_results_df = pd.DataFrame(global_results)
global_results_df.to_csv(global_output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")
print(f"Global results saved to {global_output_csv_path}")