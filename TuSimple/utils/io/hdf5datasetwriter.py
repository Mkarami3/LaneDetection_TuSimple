import h5py
import os
import json
import numpy as np
class HDF5DatasetWriter:
    
    def __init__(self, dims, outputPath, dataKey="images",bufSize=1000):
        
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
                             "exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", outputPath)
        
        #number of points to form two lanes
        number_points = 64
        print("[INFO] Writing images({}) in HDF5 format".format(dims))
        print("[INFO] Writing labels({},{}) in HDF5 format".format(dims[0],number_points))

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],number_points), dtype="float") ##????
        
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0       
        
    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
            
    
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}
        

        
    def close(self):
        
        if len(self.buffer["data"]) > 0:
            self.flush()
            
        self.db.close()
        
        
        
        
                                                
    