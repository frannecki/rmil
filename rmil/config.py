# model
BACKBONE = "resnet34"
AVGPOOL_SIZE = 1
AVGPOOL_SIZE_ATTN = 1
ATTN = False
ATTN_FEATURES = 512
PRETRAIN = False
OUT_FEATURES = 3

# training
EPOCHS = 100
LAMBDA_AUX = .5
LAMBDA_SSL = .2

# evaluation
METR = "acc"

# dataloader
NUM_CLASSES = 4
TOPK = 100
BATCH_SIZE = 16
IMAGE_SIZE = 96
SIZE = 48
PAGE = 4
IMAGE_CROP_SIZE = 96
BATCH_SIZE_AUX = 32

TRAIN_TEST_SPLIT = "data/train_test_split.json"
TRAIN_TEST_SPLIT_AUX = "data/train_test_split_aux.json"

TRAIN_LABELS_CSV = "./data/train_labels.csv"
DATA_ROOT_DIR = f"./data/mil/tiles/{SIZE}/{PAGE}"
