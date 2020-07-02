'''
- Deal with non-valid points, number of points per lane --> Done
- Normalizing labels dataset or gt labels correspond to image?! --> Done
- Writing a Video based on a new images --> Done
- Create Figure for drawing train loss, write a callback function
- How to resize picture, keep aspect ratio?
- Crop the image by the height of lane available on json file
'''
'''
0313---2858 files
0531---358 files
0601---410 files
'''
from utils.config import config
from utils.preprocessing import ImageToArrayPreprocessor
from utils.preprocessing import Preprocessor_norm,Preprocessor_resize
from utils.callback import TrainingMonitor
from utils.io import HDF5DatasetGenerator
from utils.nn import AlexNet,ShallowNet
from utils.Evaluation import evaluation_report
from utils.postprocess import PostProcessor
from utils.Augmentation import Augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
import json
import os

Training_flag = True
augmentation_flag = False
sp_img_shape = (128,128)


if augmentation_flag:
    aug1 = Augmentation(sp_img_shape)
    aug1.generate(5)
    aug1.check_code()    
    
norm   = Preprocessor_norm()
iap = ImageToArrayPreprocessor()
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, batchSize=2, KerasAPI_flag=False, preprocessors=[norm,iap],aug=None)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, batchSize=2, KerasAPI_flag=False, preprocessors=[norm,iap], aug=None)                              

if Training_flag:
    print("[INFO] compiling model...")
    opt = Adam(lr=1e-4)
    model = AlexNet.build(width=sp_img_shape[0], height=sp_img_shape[1], depth=3,lane_points=64, reg=0.0002)
    # model = ShallowNet.build(width=sp_img_shape[0], height=sp_img_shape[1], depth=3)
    model.compile(loss="mse", optimizer=opt, metrics=['mse'])
                  
    path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
    callbacks = [TrainingMonitor(path)]
    
    model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // 2,
        # steps_per_epoch=32,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // 2,
        # validation_steps=32,
        epochs=20,
        #max_queue_size=128 * 2,
        max_queue_size=10,
        callbacks=callbacks, verbose=1
        )
else:
    model = load_model("DL_model.model")

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)
trainGen.close()
valGen.close()

print("[INFO] makeing prediction on selected data set folder...")
pp = PostProcessor(new_h=sp_img_shape[1], new_w=sp_img_shape[0])
dummy = pp.make_video(model,Write_video = False)

print("[INFO] model evaluation...")
accur = evaluation_report(sp_img_shape)
accur.report(model, write_imgs = True)