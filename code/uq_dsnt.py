
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import seaborn as sns
sys.path.append("../")
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
from operator import mul
from scipy.stats import norm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from code.regression.dsnt_reg import UNetDNST
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetDNST().to(device)
model.load_state_dict(torch.load("/home/giulia/deepvalve/src/regression/regression_results/12_best_model.pth"))

dataloaders = utils.load_and_process_data(batch_size_train=8, batch_size_val_test=1, num_workers=0)

test_loader = dataloaders["test"] 



def compute_spread(heatmap):
    """Compute spread as the expected squared distance from the heatmap peak."""
    h, w = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)  # Peak location
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    squared_distances = (x_coords - peak_x) ** 2 + (y_coords - peak_y) ** 2
    spread = np.sum(heatmap * np.sqrt(squared_distances))
    return spread

def plot_heatmaps_with_spread(heatmaps, output_dir, case_id):
    """Plot heatmaps in a grid while normalizing spread values and ensuring the same color scale."""
    num_heatmaps = heatmaps.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_heatmaps)))
    
    spreads = [compute_spread(heatmaps[i].detach().cpu().numpy()) for i in range(num_heatmaps)]
    spreads = np.array(spreads)
    max_spread = np.max(spreads)
    normalized_spreads = spreads / max_spread  # Normalize spreads
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    vmin, vmax = heatmaps.min().item(), heatmaps.max().item()  # Consistent color scale
    
    for i in range(num_heatmaps):
        heatmap = heatmaps[i].detach().cpu().numpy()
        heatmap = heatmap ** 0.5  # Enhances visibility
        
        axes[i].imshow(heatmap, cmap='jet', interpolation='nearest', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"PWD: {normalized_spreads[i]:.2f}", fontsize=10)
        axes[i].axis("off")
        im = axes[i].imshow(heatmap, cmap='jet', interpolation='nearest', vmin=vmin, vmax=vmax)
    for i in range(num_heatmaps, len(axes)):
        axes[i].axis("off")

    plt.subplots_adjust(right=0.85)  
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7]) 
    plt.colorbar(im,cax= cbar_ax)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_case_{case_id}.svg"))
    plt.close()

def run_and_plot_keypoints(model, dataloader, output_dir="/home/giulia/deepvalve/uncertainty_analysis/UNET-DSNT", device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    for k, data in enumerate(dataloader):
        print(f"Processing case {k}")
      
        
        inputs = data["image"].to(device).float().unsqueeze(1)
        _, heatmaps = model(inputs)
        heatmaps = heatmaps[0]
        
        plot_heatmaps_with_spread(heatmaps, output_dir, k)


run_and_plot_keypoints(model, test_loader)
