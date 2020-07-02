from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.regularizers import l2

class Encoder:
    @staticmethod
    def build(width, height, depth,reg=0.0002):
        
        inputShape = (height, width, depth)
        
        #input layer, based on paper should be 256x480
        inputs = Input(shape=inputShape)
        chanDim = -1 # last index
        
        #First Layer
        x = BatchNormalization(axis=chanDim)(inputs)
        x = Conv2D(32, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = Conv2D(32, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        #Second Layer
        x = Conv2D(64, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = Conv2D(64, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        #Third Layer
        x = Conv2D(96, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = Conv2D(96, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        #Foruth Layer
        x = Conv2D(128, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = Conv2D(128, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        
        #Fifth Layer
        x = Conv2D(256, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        x = Conv2D(256, (3, 3), padding="same", strides=(1, 1),kernel_regularizer=l2(reg))(x)
        
        x = GlobalMaxPooling2D()(x)
        #Branch 1
        left_lane = Dense(96)(x)
        left_lane = Dense(32, name="left_ego")(left_lane)
        
        #Branch 2
        right_lane = Dense(96)(x)
        right_lane = Dense(32, name="right_ego")(right_lane)
        
        #define the model
        model = Model(
                inputs=[inputs],
                outputs=[left_lane, right_lane],
                )
        
        return model
