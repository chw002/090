# Data paths
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
DUPLICATES_CSV = 'duplicates.csv'
TRAIN_IMAGES_DIR = 'train'  # Training images directory
TEST_IMAGES_DIR = 'test'    # Test images directory

# Data processing parameters
IMAGE_SIZE = 224  # Resize images
BATCH_SIZE = 128   # Batch size
NUM_WORKERS = 16   # Number of worker threads for data loader

# Training parameters
MODEL_NAME = 'resnet50'
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda'

# Results saving
RESULTS_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

# WandB configuration
USE_WANDB = True
WANDB_PROJECT = 'ISIC2020-Classification'
WANDB_ENTITY = None
