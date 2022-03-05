# model
BACKBONE = "resnet18"
AVGPOOL_SIZE = 1
AVGPOOL_SIZE_ATTN = 1
ATTN = False
PRETRAIN = False
OUT_FEATURES = 3

# training
EPOCHS = 100
EPOCHS_SS = 50

# evaluations
METR = "acc"

# dataloader
TOPK = 100
BATCH_SIZE = 16
IMAGE_SIZE = 96
SIZE = 48
PAGE = 4
IMAGE_CROP_SIZE = 96
BATCH_SIZE_SS = 128

NUM_CLASSES = 4
TRAIN_TEST_SPLIT = "train_test_split.json"
TRAIN_TEST_SPLIT_SS = "ss/train_test_split.json"
TRAIN_LABELS_CSV = "/root/cjw/drivendata/data/csv/train_labels.csv"
DATA_ROOT_DIR = f"/guazai/drivendata/tiledata/tiles/2nd_train/{SIZE}/{PAGE}"
