import pathlib
import os

DIRECTORY_PATH = pathlib.Path(__file__).parent.resolve()
DIRECT_CONNECTION = False
DELL2_DIRECT_IP = "192.168.99.10"
DELL2_RAIL_IP = "128.32.175.252"
DELL2_IP = DELL2_DIRECT_IP if DIRECT_CONNECTION else DELL2_RAIL_IP
PI_IP = "169.254.3.203"
PABRTXL2_IP = "pabrtxl1.ist.berkeley.edu"
LOCAL = "127.0.0.1"
TRAIN_PATH = os.path.join(DIRECTORY_PATH, "template_images/train0")
TEST_PATH = os.path.join(DIRECTORY_PATH, "template_images/test0")
