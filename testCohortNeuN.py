import os
import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.metrics import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff
from datasets.segmentation_dataset import SegmentationDataset, get_train_transforms, get_val_transforms
from models.unet import UNet
from utils.losses import DiceLoss
from utils.train_eval import train_fn, eval_fn
from utils.configCohortNeuN import *

# ===== Metric functions =====
def dice_coefficient(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def iou_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)

def precision_score(pred, target):
    tp = np.logical_and(pred == 1, target == 1).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    return tp / (tp + fp + 1e-6)

def recall_score(pred, target):
    tp = np.logical_and(pred == 1, target == 1).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    return tp / (tp + fn + 1e-6)

def assd(pred, target):
    from scipy.ndimage import distance_transform_edt as edt
    pred = pred.astype(np.bool_)
    target = target.astype(np.bool_)
    dt_pred = edt(~pred)
    dt_target = edt(~target)
    sds1 = dt_target[pred]
    sds2 = dt_pred[target]
    return (sds1.mean() + sds2.mean()) / 2

def hausdorff_distance_95(pred, target):
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)
    if len(pred_points) == 0 or len(target_points) == 0:
        return np.nan
    d1 = directed_hausdorff(pred_points, target_points)[0]
    d2 = directed_hausdorff(target_points, pred_points)[0]
    return max(d1, d2)

# ===== Evaluate metrics =====
def evaluate_metrics(model, loader, device):
    model.eval()
    dices, ious, hds, precs, recs, assds = [], [], [], [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds = model(images)
            preds = torch.softmax(preds, dim=1)[:, 1, :, :].cpu().numpy()
            preds = (preds > 0.5).astype(np.uint8)

            # ===== FIX HERE: convert masks 0-255 to 0-1 =====
            masks = masks.cpu().numpy()
            masks = (masks > 127).astype(np.uint8)

            for p, m in zip(preds, masks):
                dices.append(dice_coefficient(p, m))
                ious.append(iou_score(p, m))
                hds.append(hausdorff_distance_95(p, m))
                precs.append(precision_score(p, m))
                recs.append(recall_score(p, m))
                assds.append(assd(p, m))
    
    return {
        "Dice": np.nanmean(dices),
        "IoU": np.nanmean(ious),
        "Hausdorff": np.nanmean(hds),
        "Precision": np.nanmean(precs),
        "Recall": np.nanmean(recs),
        "ASSD": np.nanmean(assds)
    }

def save_results(metrics_dict, path="results/resultExtended.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("===== Best Model Evaluation Metrics =====\n\n")
        for key, val in metrics_dict.items():
            f.write(f"{key}: {val:.4f}\n")
    print(f"📄 Results saved to {path}")

# ===== Main code =====
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(n_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

all_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))
image_paths = []
mask_paths = []
for img_path in all_images:
    filename = os.path.splitext(os.path.basename(img_path))[0]
    
    possible_masks = [
        f"{filename}_mask.png",
        f"{filename}_Mask.png",
        f"{filename} Mask.png",
        f"{filename} mask.png",
        f"{filename}_mask.tif",
        f"{filename}_Mask.tif",
        f"{filename} Mask.tif",
        f"{filename} mask.tif",
        f"{filename}Mask.png",
        f"{filename}Mask.tif",
        f"{filename}mask.png",
        f"{filename}mask.tif",
        f"{filename}_ch02_mask.png",
        f"{filename}_ch02_mask.tif",
    ]

    found_mask = None
    for mask_name in possible_masks:
        candidate_path = os.path.join(MASK_DIR, mask_name)
        if os.path.exists(candidate_path):
            found_mask = candidate_path
            break

    if found_mask:
        mask_paths.append(found_mask)
        image_paths.append(img_path)
    else:
        print(f"⚠️ No matching mask found for {filename}")
        

print("✅ Mask matching done!")

train_img, test_img, train_mask, test_mask = train_test_split(
    image_paths, mask_paths, test_size=0.15, random_state=RANDOM_SEED
)
train_img, val_img, train_mask, val_mask = train_test_split(
    train_img, train_mask, test_size=0.15, random_state=RANDOM_SEED
)
print("✅ Dataset split done!")

train_dataset = SegmentationDataset(train_img, train_mask, augment=get_train_transforms())
val_dataset = SegmentationDataset(val_img, val_mask, augment=get_val_transforms())
test_dataset = SegmentationDataset(test_img, test_mask, augment=get_val_transforms())
print("✅ Dataset objects created!")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("✅ DataLoaders ready!")

# ===== Evaluate best model =====
metrics = evaluate_metrics(model, test_loader, device)
save_results(metrics, "results/resultCohortNeuN.txt")
