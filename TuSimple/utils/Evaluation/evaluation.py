from utils.loader import DatasetLoader
from utils.config import config
import cv2
import numpy as np
import os
from statistics import mean

class evaluation_report:
    
    def __init__(self,sp_img_shape): 
        
        self.new_w = sp_img_shape[0]
        self.new_h = sp_img_shape[1]
        sdl = DatasetLoader(config,sp_img_shape)
        self.image_paths, self.labels = sdl.load(test_dataset=True) #labels are fit to processed image  
        # print("new_w={},new_h={}".format(self.new_w, self.new_h))
        
    def report(self,model, write_imgs = False):               
        
        
        for counter in range(20):#len(self.labels)
    
            label = self.labels[counter]
            img_orig = cv2.imread(self.image_paths[counter])
            
            img = cv2.resize(img_orig, (self.new_w,self.new_h),interpolation=cv2.INTER_AREA)/255.0
            img = np.expand_dims(img, axis=0)
            preds = model.predict(img)
            preds = preds.squeeze()
            error = 0
            MPE = []

            for i in range(0,64,2):
                
                x_diff = preds[i] - label[i]
                y_diff = preds[i+1] - label[i+1]
                error += np.sqrt(x_diff**2 + y_diff**2)
                
                x_gt = int(round(label[i] * (1280/self.new_w)))  
                y_gt = int(round(label[i+1] * (720/self.new_h))) 
                x_pred = int(round(preds[i] * (1280/self.new_w)))
                y_pred = int(round(preds[i+1] * (720/self.new_h)))
                
                cv2.circle(img_orig, tuple([x_gt,y_gt]), radius=5, color=(0, 255, 0),thickness = -1) #ground truth
                cv2.circle(img_orig, tuple([x_pred,y_pred]), radius=5, color=(0, 0, 255),thickness = -1) #Pred labels      
            
                if i==64:
                    break
            
            MPE.append(error/32)
            
            if write_imgs and counter <10:
                cv2.imwrite(os.path.join(config.OUTPUT_PATH , "image%04i.jpg" %counter), img_orig)
            
        print("[Info] MPE on left lane:{}".format(mean(MPE))) 
        
        
    def report_API(self,model, write_imgs = False):               
        
        
        for counter in range(20):#len(self.labels)
    
            label = self.labels[counter]
            img_orig = cv2.imread(self.image_paths[counter])
            
            img = cv2.resize(img_orig, (self.new_w,self.new_h),interpolation=cv2.INTER_AREA)/255.0
            img = np.expand_dims(img, axis=0)
            preds = model.predict(img)
            
            MPE_left_ego = []  #Mean Prediction Error
            MPE_right_ego = [] #Mean Prediction Error
            error_left_lane = 0
            error_right_lane = 0
            left_lane = preds[0].squeeze()
            right_lane= preds[1].squeeze()
            for i in range(0,32,2):
                
                x_diff = left_lane[i] - label[i]
                y_diff = left_lane[i+1] - label[i+1]
                error_left_lane += np.sqrt(x_diff**2 + y_diff**2)
                
                x_gt = int(round(label[i] * (1280/self.new_w)))  
                y_gt = int(round(label[i+1] * (720/self.new_h))) 
                x_pred = int(round(left_lane[i] * (1280/self.new_w)))
                y_pred = int(round(left_lane[i+1] * (720/self.new_h)))
                
                cv2.circle(img_orig, tuple([x_gt,y_gt]), radius=5, color=(0, 255, 0),thickness = -1) #ground truth
                cv2.circle(img_orig, tuple([x_pred,y_pred]), radius=5, color=(0, 0, 255),thickness = -1) #Pred labels      
            
                x_diff = right_lane[i] - label[i+32]  #right lane index starts from 32
                y_diff = right_lane[i] - label[i+1+32]  #right lane index starts from 32
                error_right_lane += np.sqrt(x_diff**2 + y_diff**2)   
                 
                x_gt = int(round(label[i+32] * (1280/self.new_w)))  
                y_gt = int(round(label[i+1+32] * (720/self.new_h))) 
                x_pred = int(round(right_lane[i] * (1280/self.new_w)))
                y_pred = int(round(right_lane[i+1] * (720/self.new_h)))
                
                cv2.circle(img_orig, tuple([x_gt,y_gt]), radius=5, color=(0, 255, 0),thickness = -1) #ground truth
                cv2.circle(img_orig, tuple([x_pred,y_pred]), radius=5, color=(0, 0, 255),thickness = -1) #Pred labels      
            
                if i==32:
                    break
            
            MPE_left_ego.append(error_left_lane/16)
            MPE_right_ego.append(error_right_lane/16)
            
            if write_imgs and counter <10:
                cv2.imwrite(os.path.join(config.OUTPUT_PATH , "image%04i.jpg" %counter), img_orig)
            
        print("[Info] MPE on left lane:{}".format(mean(MPE_left_ego))) 
        print("[Info] MPE on left lane:{}".format(mean(MPE_right_ego)))
