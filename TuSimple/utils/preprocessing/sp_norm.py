# import the necessary packages
import cv2

class Preprocessor_norm:
    def __init__(self):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.value = 255

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        
        image = image.astype("float") / self.value
            
        return image
    
    def get_height(self):
        return self.height
    
    def get_width(self):
        return self.width
                          