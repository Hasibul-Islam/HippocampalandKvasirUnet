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
from utils.configExtended import *

def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.tif")))

    
    mask_paths = []
    for img_path in image_paths:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Possible mask name variations
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
        else:
            print(f"⚠️ No matching mask found for {filename}")
            mask_paths.append(None)
    print("✅ Mask matching done!")

    train_img, test_img, train_mask, test_mask = train_test_split(image_paths, mask_paths, test_size=0.15, random_state=RANDOM_SEED)
    train_img, val_img, train_mask, val_mask = train_test_split(train_img, train_mask, test_size=0.15, random_state=RANDOM_SEED)
    print("✅ Dataset split done!")

    train_dataset = SegmentationDataset(train_img, train_mask, augment=get_train_transforms())
    val_dataset = SegmentationDataset(val_img, val_mask, augment=get_val_transforms())
    test_dataset = SegmentationDataset(test_img, test_mask, augment=get_val_transforms())
    print("✅ Dataset objects created!")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print("✅ DataLoaders ready!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = DiceLoss()

    print("🚀 Training started!")
    best_val_loss = float("inf")
    best_epoch = 0
    counter = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, device)
        val_loss = eval_fn(val_loader, model, loss_fn, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Model saved (Epoch {best_epoch}, Val Loss {best_val_loss:.4f})")
        else:
            counter += 1
            print(f"⚠️ No improvement. Counter {counter}/{PATIENCE}")

        if counter >= PATIENCE:
            print(f"⛔ Early stopping. Best epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}")
            break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✅ Best model loaded (Epoch {best_epoch}, Val Loss {best_val_loss:.4f})")

    test_loss = eval_fn(test_loader, model, loss_fn, device)
    print(f"🏁 Test Loss: {test_loss:.4f}")
    



    # ====== ADD THIS AT THE END OF YOUR MAIN FUNCTION ======


if __name__ == "__main__":
    main()
