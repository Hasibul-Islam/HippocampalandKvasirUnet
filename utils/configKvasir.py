import os

IMAGE_DIR = "Kvasir-SEG\\images"
MASK_DIR = "Kvasir-SEG\\masks"

MODEL_PATH = "saved_models/unet_bestKvasir.pth"

EPOCHS = 120                # small batch ধীরে শেখে → একটু বেশি epochs
BATCH_SIZE = 4
LEARNING_RATE = 1e-4        # = 2e-4 * (4/8)
PATIENCE = 20               # ধীরে improve হবে → patience বাড়াও
RANDOM_SEED = 42
USE_FP16 = True
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 1e-2
SCHEDULER = "CosineAnnealingLR"
WARMUP_EPOCHS = 5           # ছোট batch এ warmup বেশি সহায়ক
GRADIENT_CHECKPOINTING = True
GRAD_ACCUM_STEPS = 1


