'''
Thid loader returns image paths and preprocessed label dataset
used for converting images to hdf5
* not loaded into RAM!
'''
import numpy as np
import cv2
import os
import json
from itertools import chain

class DatasetLoader:
    def __init__(self, config, sp_img_shape):
        # store the image preprocessor
        self.config = config
        self.new_h = sp_img_shape[0]
        self.new_w = sp_img_shape[1]
        self.img_shape = (720,1280)
        
    def load(self, test_dataset=False):
        '''
        if loading test dataset, the flag should be true
        returns:
            image_paths: a list of paths to each individual image
            labels: (x,y) points that form the lanes
        '''
        
        #if loading test dataset, update the json file path
        if test_dataset:
            self.config.json_path = self.config.test_path
            
            
        image_paths = []
        labels  = []
        
        for folder in sorted(os.listdir(self.config.json_path)):
            
            # print('[INFO] Reading Folder named: {}'.format(folder))
            if folder.split('.')[-1] == 'json':
                json_file = self.config.json_path + "\\" + folder
                
                #if reading json file of augmented images
                if json_file.split("\\")[-1] == 'aug_lab.json': 
                    # print("json_file={}".format(json_file))
                    json_gts = [json.loads(line) for line in open(json_file)][0]
                    for json_gt in json_gts:
                        image_paths.append(json_gt['file'])
                        labels.append(json_gt['label'])
                else:    
                    json_gts = [json.loads(line) for line in open(json_file)]
                    for json_gt in json_gts:
                        
                        #Note labes are normalized to fit to image_resized 64*64
                        label = self.image_label(
                            json_gt['lanes'],json_gt['h_samples'], )
                        
                        if label == None:
                            print("[INFO] one image skipped")
                            continue
                        
                        left_lane = label[0].flatten()
                        right_lane= label[1].flatten()
                        label_flatened = np.concatenate((left_lane, right_lane),axis=None)
                        image_paths.append(self.config.json_path + json_gt['raw_file'])
                        labels.append(label_flatened) 
        
        return image_paths,labels

    def classify_lanes(self,label1,label2):
        '''
        labeling the labels and see if they are right or left lanes
        label[0] : is top point
        label[-1]: is bottom point
        '''
        pass
        
        
    def image_label(self, gt_lanes,y_samples):
        '''
        return label points normalized to (0,1)
        '''

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
                        x_norm,y_norm = self.norm_labels(List[i])
                        label.append([x_norm,y_norm]) #List[i] has two elements x,y
                
                if len(label) == 0:
                    return None
                
                label = self.label_cleaning(label)
                label_reduced.append(label)
                label = []
                
            counter += 1
                
        return label_reduced
    
    def norm_labels(self,lst,normalize=False):
        
        x = lst[0]* (self.new_h/self.img_shape[1])
        y = lst[1]* (self.new_w/self.img_shape[0])
        
        if normalize:
            x /= self.new_w
            y /= self.new_h
        
        return x,y
        
    def label_cleaning(self,label):
        '''
        receive the label points and reduce the numbers to 16 points for each lane
        '''

        label = np.array(label)
        l = []
        
        #append the first and last lane point
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

    
 