from keras.utils import np_utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize, KerasAPI_flag, preprocessors=None,aug=None): 
                 
        
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
        self.API_flag = KerasAPI_flag ##is Keras_API used 
        
    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]
                    
                if self.preprocessors is not None:
                    procImages = []
                    
                    for image in images:
                        
                        for p in self.preprocessors:
                            #already normalized by 255 when save as hdf5
                            image = p.preprocess(image) 
                            
                        procImages.append(image)
                        
                    images = np.array(procImages)
                    
                if self.aug is not None:
                    (images, labels) = next(self.aug.flow(images,labels, 
                                                          batch_size=self.batchSize))
                    
                left_ego = labels[:,0:32]
                right_ego = labels[:,32:64]
                
                if self.API_flag:
                    yield (images, {'left_ego': left_ego, 'right_ego': right_ego})
                else:
                    yield (images, labels)
                    
                # print("[info]labels shape:{}".format(labels.shape))
                # print("[info]left_ego shape:{}".format(left_ego.shape))
                
            epochs += 1
            
    def close(self):
        self.db.close()
        
        
        
        
        
        
        