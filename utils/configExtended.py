import os

IMAGE_DIR = "Images_c1_c2_660"
MASK_DIR = "Masks_c1_c2_660"

MODEL_PATH = "saved_models/unet_bestExtended.pth"

EPOCHS = 120                
BATCH_SIZE = 4
LEARNING_RATE = 1e-4        
PATIENCE = 20               
RANDOM_SEED = 42
USE_FP16 = True
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-2
SCHEDULER = "CosineAnnealingLR"
WARMUP_EPOCHS = 5           
GRADIENT_CHECKPOINTING = True
GRAD_ACCUM_STEPS = 1


