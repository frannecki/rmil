EPOCHS = 100
TOPK = 50
BATCH_SIZE = 16
# IMAGE_SIZE = 96
# IMAGE_CROP_SIZE = 96
IMAGE_SIZE = 72
IMAGE_CROP_SIZE = 72
BACKBONE = "resnet18"
# BACKBONE = "resnet_mini"

AVGPOOL_SIZE = 1
AVGPOOL_SIZE_ATTN = 1

# PAGE = 3
PAGE = 4
SIZE = 48
METR = "acc"
ATTN = False
TRAIN_TEST_SPLIT = "data/train_test_split.json"
NUM_CLASSES = 4
OUT_FEATURES = 3
# OUT_FEATURES = 1
TRAIN_LABELS_CSV = "/guazai/drivendata/data/csv/train_labels.csv"
DATA_ROOT_DIR = f"/guazai/drivendata/tiledata/tiles/2nd_train/{SIZE}/{PAGE}"

PRETRAIN = True
EPOCHS_REGION = 50
BATCH_SIZE_REGION = 128

# ANNOTATED_REGIONS_ROOT_DIR = "/mnt/data2/drivendata/annotated_regions_page4"
ANNOTATED_REGIONS_ROOT_DIR = ("/guazai/drivendata/"
                              "annotated_regions_related_0316_p4")
