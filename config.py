BATCH_SIZE = 128
SAVE_FREQ = 1
TEST_FREQ = 1
TOTAL_EPOCH = 75

# RESUME = 'model/068.ckpt'
RESUME = 'model/finetune/CASIA_B512_v2_20240728_200711/069.ckpt'
SAVE_DIR = './model/finetune'
MODEL_PRE = 'CASIA_B512_'


CASIA_DATA_DIR = 'data/CASIA'
# LFW_DATA_DIR = 'data/output'
LFW_DATA_DIR = 'data/lfw'
MY_DATA_DIR = 'data/output'

GPU = 0, 1

