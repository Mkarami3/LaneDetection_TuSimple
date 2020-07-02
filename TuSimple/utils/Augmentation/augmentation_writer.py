from utils.config import config
from utils.loader import DatasetLoader
from utils.preprocessing import Preprocessor_norm,Preprocessor_resize
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import itertools
import cv2
import random
import json



class Augmentation:
    
    def __init__(self,sp_img_shape):
        
        self.dics = []
        self.sp_img_shape = sp_img_shape
        self.resize = Preprocessor_resize(sp_img_shape[0],sp_img_shape[1]) 
    
        data_gen_args = dict(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=90,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2)
    
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)
        
        #before loading, make sure augment json file is deleted
        sdl = DatasetLoader(config,sp_img_shape)
        self.image_paths,self.labels = sdl.load() #labels are fit to processed image
        
    def generate(self, No_images = 10):
        '''
        write the augmented images to the folder config.Augmented_output
        write its labels points  in the json file
        '''
        for img_counter in range(len(self.labels)):
            path  = self.image_paths[img_counter]
            label = self.labels[img_counter]
            if img_counter == No_images:
                break
            
            image = cv2.imread(path)
            image = self.resize.preprocess(image) 
            image = img_to_array(image)
            image = np.expand_dims((image), axis=0)
            mask = np.zeros(self.sp_img_shape,dtype = 'uint8')
            mask = self.draw_circle(mask, label)        
            mask = img_to_array(mask)
            mask = np.expand_dims(mask, axis=0)
            
            seed = 1
            image_generator = self.image_datagen.flow(image,batch_size=1,seed=seed) 
            mask_generator = self.mask_datagen.flow(mask,batch_size=1,seed=seed)  
        
            self.write_augm_images(img_counter,image_generator)
            self.write_augm_masks(img_counter, mask_generator)
            
            if img_counter%50 == 0:
                print("[INFO] 50 new augmented images added...")
                
        self.write_json(self.dics)#save json file

    def draw_circle(self,mask,label):
        '''
        draw label points on the mask, before augmentation 
        '''

        for i in range(0,64,2):
            x = int(round(label[i]))
            y = int(round(label[i+1]))
            cv2.circle(mask, tuple([x,y]), radius=1, color=255,thickness = -1)
            if i==64:
                break 
            
        return mask
    
    
    def write_augm_images(self,img_counter,image_generator):
        '''
        write the augmeneted images in the address config.Augmented_output
        '''
        total = 0
        for img_gen in image_generator:

            img_filename = config.Augmented_output + "img_"+str(img_counter)+"_aug_%04i.jpg" %total
            cv2.imwrite(img_filename, img_gen[0,:,:,:])
            total += 1
            if total == 5:
                break  
    def write_augm_masks(self,img_counter, mask_generator):
        total = 0
        for mask_gen in mask_generator:
            img_filename = config.Augmented_output + "img_"+str(img_counter)+"_aug_%04i.jpg" %total
            # cv2.imwrite(img_filename, mask_gen[0,:,:,:])
            label_flat = self.find_label_points(mask_gen)

            D = {"label": label_flat, "file": img_filename}
            self.dics.append(D)
            
            total += 1
            if total == 5:
                break     
            
    def find_label_points(self, mask_gen):
        '''
        find x,y pixel coordinates of label points in the mask file
        '''

        mask_gen = mask_gen[0,:,:,0].astype('uint8')
        (thresh, im_bw) = cv2.threshold(mask_gen, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        indices = []
        for i in range(self.sp_img_shape[0]):
            for j in range(self.sp_img_shape[1]):
                
                if im_bw[i,j] == 255:
                    flag = self.unique_label(j,i,indices)
                    if flag: 
                        indices.append([j,i]) 

        # print("number of points before filtering={}".format(len(indices)))
        # label points should be 32, if any extra should be filter out
        extra_points = len(indices) - 32 

        for i in range(np.abs(extra_points)):
            index = random.randint(0, 32)
            if extra_points > 0:
                indices.pop(index)
            else:
                indices.append(indices[0])
        
        # print("number of points after filtering={}".format(len(indices)))
        label_flat = list(itertools.chain(*indices))
        return label_flat    

        
    def unique_label(self,j,i,indices):
        
        for count in range(len(indices)):
            dist = np.sqrt(((indices[count][0] - j)**2 + (indices[count][1] - i)**2))
            if dist < 2.3:
                # print("i={},j={},min_dis={}".format(i,j,dist))
                return False
            
        return True       
    
    
    def write_json(self,dics):
        '''
         save the labels and image file names as json file
        '''
        f = open(config.LABEL_AUGM_PATH, "w")
        f.write(json.dumps(dics))
        f.close()   


    def check_code(self):
        '''
        checking the above code
        '''
        json_file = "D:\\NRC\\LaneDetection\\TuSimple\\training_dataset\\aug_lab.json"
        json_gts = [json.loads(line) for line in open(json_file)][0]
        
        image_paths = []
        labels  = []
        for json_gt in json_gts:
            image_paths.append(json_gt['file'])
            labels.append(json_gt['label'])

        img_test = image_paths[20] 
        img = cv2.imread(img_test)
        label_test= labels[20] 
        
        for i in range(0,64,2):
            x = label_test[i]
            y = label_test[i+1]  
            cv2.circle(img, (x,y), radius=1, color=(0, 255, 0))    
            if i==64:
                break
        
        
        cv2.imshow("", img)

    