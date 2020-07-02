# import the necessary packages
from utils.config import config
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

test_path = 'D:\\NRC\\LaneDetection\\TuSimple\\utils\\DataSet_selected\\'

class PostProcessor:
    def __init__(self, new_h, new_w):
        # store the target image width, height, and interpolation
        # method used when resizing
        # self.trainX =  trainX
        # self.trainY =  trainY
        # self.testX  = testX
        # self.testY  = testY
        # self.predictions  = predictions
        # self.label_path = label_path
        # self.image_path = image_path
        # self.idx_testX = idx_testX
        self.new_h = new_h
        self.new_w = new_w
    
    def make_video(self,model, Write_video = False):
        
        counter = 0
        for folder in sorted(os.listdir(test_path)):
            
            # print(["[INFO] Reading Folder named: {}".format(folder)])
            image_path=os.listdir(test_path + folder)
            
            for image in image_path:
        
                img = cv2.imread(test_path + folder +'\\'+ image)
        
                image1 = cv2.resize(img,(self.new_w,self.new_h),interpolation=cv2.INTER_AREA)/255.0
                image1 = np.expand_dims(image1, axis=0)
                pred_test = model.predict(image1)
                
                for lane in pred_test:
                    lane = lane.squeeze()
                    for i in range(0,len(lane),2):
                        x = int(round(lane[i] * (1280/self.new_w)))
                        y = int(round(lane[i+1] * (720/self.new_h)))
                        cv2.circle(img, tuple([x,y]), radius=5, color=(0, 0, 255),thickness = -1)
                        if i==64:
                            break
                
                cv2.imwrite(os.path.join(config.OUTPUT_PATH , "image%04i.jpg" %counter), img)
                counter += 1
        
        if Write_video:
            print("[INFO] Writing a Video...")
            img_array = []
            for filename in glob.glob(config.OUTPUT_PATH  + '\\' +  '*.jpg'):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
             
             
            out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
             
            for i in range(len(img_array)):
                out.write(img_array[i])
                
            out.release()

    # def vis(self, idx):


    #     json_gt = [json.loads(line) for line in open(self.label_path)]
        
    #     index = self.idx_testX[idx]
        
    #     gt = json_gt[index]
    #     gt_lanes = gt['lanes']
    #     y_samples = gt['h_samples']
    #     raw_file = gt['raw_file']
    #     gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples)] for lane in gt_lanes]
        
    #     image = cv2.imread(self.image_path+raw_file)
        
    #     counter  = 0
    #     for lane in gt_lanes_vis:
            
    #         if counter == 0 or counter == 1: 
    #             # print("length of pts={}".format(len(lane)))
    #             for pt in lane:
    #                 if pt[0]>0:
    #                     x = int(pt[0])
    #                     y = int(pt[1])
    #                     # print("x={},y={}".format(x,y))
    #                     cv2.circle(image, (x,y), radius=5, color=(0, 255, 0))
                    
    #         counter += 1
        
    #     pred_test = self.predictions[idx]
    #     for i in range(0,len(pred_test),2):
    #         x = int(round(pred_test[i] * (1280/self.new_w)))
    #         y = int(round(pred_test[i+1] * (720/self.new_h)))
    #         cv2.circle(image, tuple([x,y]), radius=5, color=(0, 0, 255),thickness = -1)
    #         if i==64:
    #             break
            
    #     cv2.imshow("test",image)
    #     cv2.imwrite("image%04i.jpg" %idx, image)
        
    # def accuracy(self):
    #     '''
    #     return mse between predictions and test data set
    #     '''

    #     accuracy = []         
    #     for i in range(self.predictions.shape[0]):     
    #         for j in range(0,self.predictions.shape[1], 2):
    #             predX = int(round(self.predictions[i,j] * (1280/self.new_w)))
    #             predY = int(round(self.predictions[i,j+1] * (720/self.new_h)))
                
    #             gtX = int(round(self.testY[i,j] * (1280/self.new_w)))
    #             gtY = int(round(self.testY[i,j+1] * (720/self.new_h)))
                
    #             diff = np.sqrt(np.power(gtX-predX,2) + np.power(gtY-predY,2))
                
    #             accuracy.append(diff)
    #             if j==64:
    #                 continue
                
            
    #     mse = np.mean(np.array(accuracy))
        
    #     print("[INFO]Mean Squarred Error={}".format(mse))