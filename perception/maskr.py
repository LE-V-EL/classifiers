import os,sys,time

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/mask_rcnn/'))

# Mask R-CNN
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn import utils
from mrcnn import visualize

import tensorflow as tf

from . import config as C

class MaskR:

    def __init__(self, name='perception', magicnumber=0, init_with='coco'):

        t0 = time.time()

        print ('GPU available:', tf.test.is_gpu_available())


        self.mode = 'training'
        self.config = C.TrainingConfig()
        self.weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
        self.model_dir = os.path.join(self.weights_dir, name)

        if os.path.exists(self.model_dir+str(magicnumber)):
            # we need to increase the magicnumber until we find a good one
            while os.path.exists(self.model_dir+str(magicnumber)):
                magicnumber += 1

        print ('Storing in ', self.model_dir+str(magicnumber))
        os.mkdir(self.model_dir+str(magicnumber))

        self.model = MaskRCNN(self.mode, self.config, self.model_dir)

        # Which weights to start with?
        # imagenet, coco, or last

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(self.weights_dir, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)        

        if init_with == "imagenet":
            self.model.load_weights(self.model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            self.model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            self.model.load_weights(model.find_last(), by_name=True)


        self.testModel = None


        print ('Setup complete after', time.time()-t0, 'seconds')



    def train(self, dataset_train, dataset_val, epochs=1):
        '''
        '''
        t0 = time.time()

        self.model.train(dataset_train, dataset_val,
                         learning_rate=self.config.LEARNING_RATE,
                         epochs=epochs,
                         layers='heads')


        print ('Training complete after', time.time()-t0, 'seconds')



    def predict(self, image, verbose=True):
        '''
        '''
        if not self.testModel:

            model = MaskRCNN(mode="inference", 
                              config=C.TestingConfig(),
                              model_dir=self.model_dir)

            model.load_weights(model.find_last(), by_name=True)

            self.testModel = model

        results = self.testModel.detect(image)

        if verbose:
            r = results[0]
            visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], 
                                        ["",""], r['scores'],figsize=(10,10))

        return results
