'''
Load images and apply preprocess algorithms on them
Author: Mohammad Karami
Date: April, 2020
Acknowledgement:
    Joseph Hows and Joe MiniChino, "Learning OpenCV4"
    Adrian Rosebrock, "PyimageSearch"
'''
import numpy as np
import pandas as pd
import cv2
import os
import json
import random
from itertools import chain

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []
            self.new_h = 64
            self.new_w = 64
        else:
            #preprocessors[0] is SimplePreprocessor
            self.new_h = preprocessors[0].get_height() 
            self.new_w = preprocessors[0].get_width()

    def load(self, image_path, LabelPath, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        
        json_gt = [json.loads(line) for line in open(LabelPath)]
        
        counter=0
        indices = []
        # print("length_json={}".format(len(json_gt)))
        # loop over the input images
        for (i, gt) in enumerate(json_gt):
            counter += 1
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            raw_file = gt['raw_file']
        
            image = cv2.imread(image_path+raw_file)
            image_shape = image.shape
            label = self.image_label(gt_lanes,y_samples, image_shape)
            
            if label == None:
                print("[INFO] one image skipped")
                continue
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
        
            data.append(image)
            labels.append(label)
            indices.append(counter-1)
            
            # print("ii={}".format(ii))
            
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}".format(i + 1))
        
        labels = np.array(labels).reshape(-1,64)
        return (np.array(data), np.array(labels),indices,image_shape[1]/self.new_h, image_shape[0]/self.new_w)
    
    
    def image_label(self, gt_lanes,y_samples, img_shape):
        
        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples)] for lane in gt_lanes]
        
        label = []
        #reduce number of points in each lane
        #32 nested list or 16 points (x,y) for each lane
        label_reduced = []
        counter  = 0
        for List in gt_lanes_vis:
            
            if counter == 0 or counter == 1: #just consider the ego lanes
                
                #for skipping two points on the lane and reducing the lane length
                #from 48 to 16
                for i in range(len(List)):
                    
                    if List[i][0] > 0:
                        x = List[i][0]* (self.new_h/img_shape[1])
                        y = List[i][1]* (self.new_w/img_shape[0])
                        label.append([x,y]) #List[i] has two elements x,y
                
                if len(label) == 0:
                    return None
                
                label = self.label_cleaning(label)
                label_reduced.append(label)
                label = []
                
            counter += 1
        
        
        # label_reduced = list(chain.from_iterable(label_reduced))
        return label_reduced
    
    def label_cleaning(self,label):
        label = np.array(label)
        l = []
        l.append(label[0])
        l.append(label[-1])
        step = label.shape[0]//14
        index = 0
        
        for i in range(14):
            index += step
            if index < label.shape[0]:
                l.append(label[index])
            else:
                l.append(label[-2])
            
        return np.array(l) 

    
 