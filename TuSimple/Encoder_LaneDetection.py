'''
This architecture is based on the paper:
Reliable multilane detection and classification by utilizing CNN as a regression network, Chougule et al. (2018)
- Cannot handle lane changes!
'''

from utils.config import config
from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import Preprocessor_norm,Preprocessor_resize
from utils.callback import TrainingMonitor
from utils.io import HDF5DatasetGenerator
from utils.nn import Encoder
from utils.Evaluation import evaluation_report
from utils.postprocess import PostProcessor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import json
import os

Training_flag = False  # Want to train or load already trained model
sp_img_shape = (480,256) #((width, height)

norm   = Preprocessor_norm()
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batchSize=100,preprocessors=[norm,iap], aug=None)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batchSize=100,preprocessors=[norm,iap], aug=None)                              

if Training_flag:
    print("[INFO] compiling model...")

    
    opt = Adam(lr=1e-4)
    model = Encoder.build(width=sp_img_shape[0], height=sp_img_shape[1], depth=3)
    model.compile(optimizer=opt, loss={'left_ego': 'mean_squared_error', 'right_ego': 'mean_squared_error'})
    model.summary()
                  
    path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
    callbacks = [TrainingMonitor(path)]
    
    
    model.fit_generator(
            trainGen.generator(),
            # steps_per_epoch=trainGen.numImages // 16,
            steps_per_epoch=40,
            validation_data=valGen.generator(),
            # validation_steps=valGen.numImages // 16,
            validation_steps=40,
            epochs=10,
            #max_queue_size=128 * 2,
            max_queue_size=10,
            callbacks=callbacks, verbose=1
            )
    
    print("[INFO] serializing model...")
    model.save(config.MODEL_PATH, overwrite=True)
    trainGen.close()
    valGen.close()
else:
    model = load_model("DL_model.model")
    

print("[INFO] model evaluation...")
accur = evaluation_report(sp_img_shape)
accur.report(model, write_imgs = True)

print("[INFO] making prediction...")
pp = PostProcessor(new_h=sp_img_shape[1], new_w=sp_img_shape[0])
dummy = pp.make_video(model,Write_video = False)
