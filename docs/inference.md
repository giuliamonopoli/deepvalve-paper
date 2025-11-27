# Running Inference with DeepValve

This page demonstrates how to run the full DeepValve inference pipeline on a single cardiac MRI:
1. Automatically detect a bounding box around the heart using a pre-trained YOLO model.
2. Use the DSNT-based keypoint model to predict valve landmarks inside this box.

## 1. Overview of the demo script

The demo script does the following:

1. **Load a single image** (DICOM or NIfTI) from disk.
2. **Normalize and convert** the image to the format expected by YOLO.
3. **Run YOLO** to obtain a bounding box around the heart.
4. **Crop the original image** to the detected box, with safe padding if needed.
5. **Preprocess the crop** (resize, normalize) for the DSNT keypoint model.
6. **Run the DSNT model** to obtain septal and lateral valve keypoints.
7. **Map keypoints back to image coordinates** and **visualise** them overlaid on the crop.

The code below is a complete, runnable example. You mainly need to adjust the paths under the `# --- EDIT THESE ---` section.


##  Step 1 – Automatic bounding-box detection (YOLO)

- The image is loaded from `DICOM_PATH` (DICOM or NIfTI).
- `YOLO_WEIGHTS` points to the bounding box model trained on heart regions.
- We take the highest‑confidence detection from `res.boxes`.

This gives us a bounding box `[x1, y1, x2, y2]` around the heart.

##  Step 2 – Keypoint prediction (DSNT model)

- The grayscale image is cropped to this bounding box.
- The crop is resized to `TARGET_SIZE` and normalised.
- We pass it through the DSNT model.
- The output tensor is split into:
  - **Septal** landmarks (first 10 points)
  - **Lateral** landmarks (last 10 points)

Finally, we map these keypoints back to the crop coordinates using
`utils.readjust_keypoints` and overlay them in red (septal) and blue (lateral).

## 3. Full inference demo script

```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import torch
from ultralytics import YOLO
import albumentations as A
import sys
import nibabel as nib

sys.path.append("../")  
import utils
from dsnt_model import UNetDSNT

# --- EDIT THESE ---
DICOM_PATH = "/MnM2/dataset/274/274_LA_ES.nii.gz"
YOLO_WEIGHTS = "./bbx_model.pt"
KP_MODEL_PATH = "./dsnt_best_model.pth"
# -------------------
TARGET_SIZE = (448, 448)  
DEVICE = torch.device("cpu")


def load_dicom_image(path, frame_idx=0):
    """Load a single 2D frame from a DICOM or NIfTI file as float32."""
    ext = os.path.splitext(path)[-1].lower()
    if ext not in [".dcm", ".gz", ".nii"]:
        raise ValueError(f"Unsupported file extension: {ext}")
    elif ext == ".dcm":
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        # Apply rescale slope/intercept if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = arr * slope + intercept

        # Handle MONOCHROME1 if needed
        pi = getattr(ds, "PhotometricInterpretation", "").upper()
        if "MONOCHROME1" in pi:
            img = img.max() - img

        return img.astype(np.float32)
    elif ext == ".nii" or ext == ".gz":
        nifti_file = nib.load(path)
        img = nifti_file.get_fdata()[:, :, frame_idx]
        return img.astype(np.float32)

def scale_to_uint8(img):
    """Scale a float32 image to the 0–255 uint8 range."""
    img = img.astype(np.float32)
    img = img - float(img.min())
    mx = float(img.max())
    if mx > 0:
        img = (img / (mx + 1e-8) * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img

def ensure_bgr_for_yolo(gray_uint8):
    """YOLO expects 3 channels; convert grayscale -> BGR."""
    return cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2BGR)

def safe_crop_with_pad(img, x0, y0, x1, y1):
    """Crop [x0, y0, x1, y1] from img, padding if the box goes out of bounds."""
    H, W = img.shape[:2]
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - W)
    pad_bottom = max(0, y1 - H)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        img = np.pad(
            img,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top

    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = max(x0 + 1, int(x1))
    y1 = max(y0 + 1, int(y1))
    return img[y0:y1, x0:x1]

def prepare_for_kp_model(crop_gray, target_size=TARGET_SIZE):
    """Resize and normalize the cropped image for the DSNT keypoint model."""
    resized = cv2.resize(
        crop_gray, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC
    )

    if resized.dtype != np.uint8:
        resized = scale_to_uint8(resized)

    img_ch = resized[..., None]
    preproc = A.Compose(
        [A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0)]
    )
    out = preproc(image=img_ch)
    img_norm = out["image"].astype(np.float32)  # HxWx1 float32

    tensor = (
        torch.from_numpy(img_norm)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )
    return tensor, resized

def run():
    # 1) Load image
    img = load_dicom_image(DICOM_PATH)  # HxW float32
    img8 = scale_to_uint8(img)         # HxW uint8

    # 2) YOLO bounding-box detection
    yolo_in = ensure_bgr_for_yolo(img8)  
    yolo_model = YOLO(YOLO_WEIGHTS)
    results = yolo_model.predict(
        source=yolo_in, conf=0.25, iou=0.5, save=False, verbose=False
    )
    if not results:
        raise RuntimeError("YOLO returned no results")

    res = results[0]
    if not hasattr(res, "boxes") or len(res.boxes) == 0:
        raise RuntimeError("No boxes detected by YOLO")

  
    xyxy = res.boxes.xyxy.cpu().numpy() 
    confs = res.boxes.conf.cpu().numpy()
    # choose box with highest confidence
    idx = int(np.argmax(confs))
    x1, y1, x2, y2 = map(int, xyxy[idx].tolist())

    # 3) Crop original grayscale image 
    crop = safe_crop_with_pad(img, x1, y1, x2, y2) 
    if crop.size == 0:
        raise RuntimeError("Empty crop after safe_crop")

    # 4) Prepare crop for keypoint model
    inp_tensor, resized_for_vis = prepare_for_kp_model(crop, TARGET_SIZE)

    # 5) Load keypoint model and run
    kp_model = UNetDNST().to(DEVICE)
    kp_model.load_state_dict(torch.load(KP_MODEL_PATH, map_location=DEVICE))
    kp_model.eval()
    with torch.no_grad():
        outputs = kp_model(inp_tensor)  # expected shape (1, 20, 2)

    # 6) Split septal/lateral and readjust to image coordinates
    original_size = inp_tensor.shape[-2:]  # (H, W)
    outputs = outputs.cpu().numpy()
    outputs_septal = outputs[:, :10, :]    # (1, 10, 2)
    outputs_lateral = outputs[:, 10:, :]   # (1, 10, 2)

    outputs_septal = utils.readjust_keypoints(outputs_septal, original_size)
    outputs_lateral = utils.readjust_keypoints(outputs_lateral, original_size)

    # 7) Plot results on resized crop
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(resized_for_vis, cmap="gray")
    for k in range(outputs_septal.shape[1]):
        ax.plot(
            outputs_septal[0, k, 0],
            outputs_septal[0, k, 1],
            "ro",
            markersize=4,
        )
    for k in range(outputs_lateral.shape[1]):
        ax.plot(
            outputs_lateral[0, k, 0],
            outputs_lateral[0, k, 1],
            "bo",
            markersize=4,
        )
    ax.set_title(f"Detected bbox: [{x1},{y1},{x2},{y2}]")
    ax.axis("off")
    plt.show()

    # Return detection and keypoints
    return dict(
        bbox=(x1, y1, x2, y2),
        septal=outputs_septal[0].tolist(),
        lateral=outputs_lateral[0].tolist(),
    )

if __name__ == "__main__":
    out = run()
    print("Result:", out)