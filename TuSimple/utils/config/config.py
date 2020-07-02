import os.path
my_path = os.path.dirname(__file__)

json_path = "D:\\NRC\\LaneDetection\\TuSimple\\training_dataset\\"
test_path = "D:\\NRC\\LaneDetection\\TuSimple\\test_dataset\\"

LABEL_AUGM_PATH = "D:\\NRC\\LaneDetection\\TuSimple\\training_dataset\\aug_lab.json"
Augmented_output = "D:\\NRC\\LaneDetection\\TuSimple\\training_dataset\\clips\\Augmented\\"

NUM_VAL_IMAGES = 720
NUM_TEST_IMAGES = 300

TRAIN_HDF5 = "D:\\NRC\\LaneDetection\\TuSimple\\utils\\HDF5\\train.hdf5"
VAL_HDF5 = "D:\\NRC\\LaneDetection\\TuSimple\\utils\\HDF5\\val.hdf5"
TEST_HDF5 = "D:\\NRC\\LaneDetection\\TuSimple\\utils\\HDF5\\test.hdf5"

MODEL_PATH = "D:\\NRC\\LaneDetection\\TuSimple\\utils\\HDF5\\DL_model.model"
DATASET_MEAN = "D:\\NRC\\LaneDetection\\TuSimple\\utils\\HDF5\\RGB_mean.json"

# define the path to the output directory used for storing plots,
OUTPUT_PATH = "D:\\NRC\\LaneDetection\\TuSimple\\utils\\output\\"
