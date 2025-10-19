import os

IMAGE_DIR = "C:\\PythonProjects\\PersonalProjects\\PiashVai\\KvasirSeg\\Kvasir-SEG\\Kvasir-SEG\\images"
MASK_DIR = "C:\\PythonProjects\\PersonalProjects\\PiashVai\\KvasirSeg\\Kvasir-SEG\\Kvasir-SEG\\masks"

MODEL_PATH = "saved_models/unet_bestKvasir.pth"

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


