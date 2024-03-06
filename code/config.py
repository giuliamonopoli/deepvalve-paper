# config.py
import random

## Augmentation parameters

rotation_angle = random.randint(20, 80)

brightness_limit = [0.01, 0.15]
contrast_limit = [0.1, 0.2]

# Annotation data paths
data_frame_path = "/home/daniel/deepvalve/data/data_new.csv"

# original images path of patients images
raw_imgs_path = "/path/to/raw/images"


# export noraml annotations for leaflets
annotation_folder = (
    "/Users/giuliamonopoli/Desktop/PhD /deepvalve/AW_MAD-redo_NF_20231121"
)

