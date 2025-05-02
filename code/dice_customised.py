from skimage.morphology import skeletonize, skeletonize_3d
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils
from matplotlib.ticker import FormatStrFormatter
from  segmentation_models_pytorch.utils.metrics import Fscore, IoU
import numpy as np
from scipy.ndimage import binary_dilation
new_seg = torch.load("path/to/your/segmentation_results.pth")  # Replace with your actual path
new_images = new_seg['images']
new_pred = new_seg['predicted_mask']
new_g_t_mask = new_seg['ground_truth_mask']


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def dice_coefficient(y_true, y_pred):
    """Compute the Dice coefficient for binary masks."""
    f = Fscore()
    y_true_f = utils.reverse_one_hot(predmask_list[i])
    y_pred_f = utils.reverse_one_hot(gt_mask_list[i])
    
    return f(y_true_f, y_pred_f)
def dilated_dice(true_mask, pred_mask):
    """
    Calculate the Dilated Dice coefficient between the true and predicted binary masks.
    
    :param true_mask: Ground truth binary mask as a numpy array.
    :param pred_mask: Predicted binary mask as a numpy array.
    :param dilation_radius: The radius of the structuring element for the dilation.
    :return: The Dilated Dice coefficient.
    """
 
    structuring_element = np.ones((2,2), dtype=bool)

    # Dilate the true and predicted masks
    dilated_true_mask = binary_dilation(true_mask, structure=structuring_element)
    dilated_pred_mask = binary_dilation(pred_mask, structure=structuring_element)

    # Calculate intersections
    inter_true_dilated_pred = np.logical_and(true_mask, dilated_pred_mask).sum()
    inter_pred_dilated_true = np.logical_and(pred_mask, dilated_true_mask).sum()

    true_sum = np.sum(true_mask)
    pred_sum = np.sum(pred_mask)

    # Calculate the Dilated Dice coefficient
    dilated_dice_coeff = (inter_true_dilated_pred + inter_pred_dilated_true) / (true_sum + pred_sum)

    return dilated_dice_coeff


Ddice=[]
for i,j in enumerate(new_pred):
    mask_pred = np.array(j).astype(float)
    mask_pred =  np.argmax(mask_pred, axis=-1)
    mask = np.array(new_g_t_mask[i]).astype(float)
    mask =  np.argmax(mask, axis=-1)
    ddice = dilated_dice(mask, mask_pred)
    Ddice.append(ddice)
cldice =[]
for i,j in enumerate(new_pred):
    mask_pred = np.array(j).astype(float)
    mask_pred =  np.argmax(mask_pred, axis=-1)
    mask = np.array(new_g_t_mask[i]).astype(float)
    mask =  np.argmax(mask, axis=-1)
    cldice.append(clDice(mask_pred,mask))
dice = []
for i,j in enumerate(new_pred):
    dice_coeff = dice_coefficient(new_g_t_mask[i], new_pred[i])
    dice.append(dice_coeff)

data = np.array([dice,cldice,Ddice]).T



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

xplot = [0, 1, 2]
idx_list = [2,11,9,14]

# >>>>> HERE YOU SPECIFY IDX TO CORRESPOND TO YOUR EXAMPLE CASES
idx_examples = {idx: color for idx, color in zip(idx_list, colors[:4])}

fig_width_cm = 50/4  # Width in centimeters
fig_height_cm = 30/4 # Height in centimeters

# # Convert centimeters to inches
fig_width_in = fig_width_cm / 2.54
fig_height_in = fig_height_cm / 2.54

fig, ax = plt.subplots(constrained_layout=True, figsize=(fig_width_in, fig_height_in))
for i in range(data.shape[0]):
    if i in idx_examples:
        ax.plot(xplot, data[i, :],
                '.-', c=idx_examples[i],
                linewidth=2.0,
                alpha=1.0,
                label=f"Case {i}",
                )

    ax.plot(xplot, data[i, :],
            '.-', c='k',
            linewidth=0.8,
            alpha=0.3,
            label=None,
            )

for j in range(data.shape[1]):
    sns.boxplot(x=j, y=data[:, j], ax=ax, linewidth=1.0, fliersize=0.0, width=0.5, zorder=0, color="0.8")

ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Dice", "CDice", "DDice"], fontdict={'fontsize': 8})
ax.set_ylabel("Metric value",fontdict={'fontsize': 8})
ax.set_yticklabels(ax.get_yticks(), fontdict={'fontsize': 8})
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel("Metrics",fontdict={'fontsize': 8})
ax.legend().set_visible(False)
plt.tight_layout()
plt.savefig("metrics.svg")