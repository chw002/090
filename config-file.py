# 项目配置文件

# 数据路径
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
DUPLICATES_CSV = 'duplicates.csv'
TRAIN_IMAGES_DIR = 'train_images'  # 训练图像目录
TEST_IMAGES_DIR = 'test_images'    # 测试图像目录

# 数据处理参数
IMAGE_SIZE = 224  # 调整图像大小，较小的尺寸训练更快
BATCH_SIZE = 32   # 批次大小
NUM_WORKERS = 8   # 数据加载器的工作线程数

# 训练参数
MODEL_NAME = 'resnet50'  # 可选: 'alexnet', 'vgg16', 'resnet50', 'efficientnet_b0', 'mobilenet_v2', 'vit'
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = 'cuda'  # 'cuda' 或 'cpu'

# 结果保存
RESULTS_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

# WandB配置
USE_WANDB = True
WANDB_PROJECT = 'ISIC2020-Classification'
WANDB_ENTITY = None  # 你的WandB账户名
