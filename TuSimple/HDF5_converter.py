from utils.config import config
from sklearn.model_selection import train_test_split
from utils.preprocessing import Preprocessor_norm,Preprocessor_resize
from utils.io import HDF5DatasetWriter
from utils.loader import DatasetLoader
import numpy as np
import progressbar
import json
import cv2
import os


sp_img_shape = (224,224)
resize = Preprocessor_resize(sp_img_shape[0],sp_img_shape[1]) 

sdl = DatasetLoader(config,sp_img_shape)
image_paths,labels = sdl.load() #labels are fit to processed image

(trainPaths, testPaths, 
 trainLabels, testLabels) = train_test_split(
                             image_paths, labels, 
                             test_size=config.NUM_TEST_IMAGES,random_state=42)
                         
(trainPaths, valPaths, 
 trainLabels, valLabels)  = train_test_split(trainPaths, trainLabels,
                        test_size=config.NUM_VAL_IMAGES,random_state=42) 
                                             

datasets = [
("train", trainPaths, trainLabels, config.TRAIN_HDF5),
("val", valPaths, valLabels, config.VAL_HDF5),
("test", testPaths, testLabels, config.TEST_HDF5)]

(R, G, B) = ([], [], [])

for (dType, paths, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), sp_img_shape[1],sp_img_shape[0], 3), outputPath)
    
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    
    pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()
    
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        image = cv2.imread(path)
        image = resize.preprocess(image) 
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        writer.add([image], [label])
        pbar.update(i)
    
    pbar.finish()
    writer.close()

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()     




                               