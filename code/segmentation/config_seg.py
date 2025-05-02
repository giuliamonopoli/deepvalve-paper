import segmentation_models_pytorch.utils as smp_utils
import torch

## dataset parameters
raw_imgs_path = "/path/to/raw/images"
annotation_json = "/path/to/annotations.json"

## these two not needed if the annotations.json file is in the same folder
annotation_folder = "/path/to/annotations.json" 
data_frame_path = "/path/to/data.csv"

labels = "label_class_dict.csv"

# create_data parameters (this is for creation only and not for giving input to model for that use model parameters)
no_of_points = 30  # arbitrary number of points to be used for creating the mask
mask_thickness = 2  # arbitrary mask thickness to be used for creating the mask
path_for_segmentation_data_folder = f"."  ## base path for all the segmentation data
mask_generation_method = "line"  # 'polygon' or 'line'

## model parameters

## if True then augmentations are applied can repeat the same number for multiple times as the probability of
# that augmentation is not 1 so random images will be augmented
augmentation = 3

n_points = 30
w_mask = 2

# Important! note the path below assumes the masks are already created and saved in the folder with that name
path_for_segmentation_data_model = f"../../data/segmentation_data/segmentation_lines_{n_points}_{w_mask}"  ## path for one of the trail,test,val folders to used in model
run_name = "your_run"  ## name of the run for saving the model , by uses the random run name from weights and biases
seed = 42
ENCODER = "efficientnet-b4"
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = "softmax2d"
threshold = 0.6  # threshold for IOU score

data = {
    "mask_thickness": path_for_segmentation_data_model.split("_")[-1],
    "no_of_points": path_for_segmentation_data_model.split("_")[-2],
    "mask_generation_method": mask_generation_method,
}

## training parameters
device = "cuda:1" if torch.cuda.is_available() else "mps"
batch_size = 8
epochs = 1000
workers = 0
patience = 500
metrics = smp_utils.metrics.IoU(threshold=threshold)
lr_scheduler_step_size = 20  # how many epochs the learning rate will be decayed
lr_scheduler_gamma = 0.3  # how much the learning rate will be reduced after each step

scheduler_factor = 0.91
base_lr = 0.001
max_lr = 0.01
scheduler_patience = 8
mode = "triangular2"
scheduler_threshold = 0.018
step_size_div_factor = 2  ## the number to divide step size by

##test_data parameters
x_test_dir = path_for_segmentation_data_model + "/test/images/"
y_test_dir = path_for_segmentation_data_model + "/test/masks/"

## optimizer parameters
lr = 0.04  ## learning rate for adam optimizer
weight_decay = 0
