# import the necessary packages
import cv2

class Preprocessor_resize:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        
        image = cv2.resize(image, (self.width, self.height),interpolation=self.inter)
        return image
    
    def get_height(self):
        return self.height
    
    def get_width(self):
        return self.width
                          