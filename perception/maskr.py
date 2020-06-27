import os,sys,time

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), '../external/mask_rcnn/'))
import tensorflow as tf

tf.keras
# Mask R-CNN
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn import utils
from mrcnn import visualize
from keras.callbacks import History

import tensorflow as tf
import pickle     as p

import perception.config as C

class MaskR:

    def __init__(self, model_dir, init_with='coco'):

        t0 = time.time()

        print ('GPU available:', tf.test.is_gpu_available())


        self.mode = 'training'
        self.config = C.TrainingConfig()
        self.model_dir = model_dir

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        print ('Storing in ', self.model_dir)

        self.model = MaskRCNN(self.mode, self.config, self.model_dir)

        # Which weights to start with?
        # imagenet, coco, or last

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(self.model_dir, "mask_rcnn_coco.h5")
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


        print ('MaskRCNN Setup complete after', time.time()-t0, 'seconds')



    def train(self, dataset_train, dataset_val, epochs):
        '''
        '''
        t0 = time.time()

        history = History()

        self.model.train(dataset_train, dataset_val, custom_callbacks=[history],
                         learning_rate=self.config.LEARNING_RATE,
                         epochs=epochs,
                         layers='heads')

        p.dump(history.history, open(os.path.join(self.model_dir, "history.p"), "wb"))

        print ('MaskRCNN Training complete after', time.time()-t0, 'seconds')

        return history



    def predict(self, images, verbose=False, weights_path=None):
        '''
        '''

        t0 = time.time()

        if not self.testModel:

            model = MaskRCNN(mode="inference", 
                              config=C.TestingConfig(),
                              model_dir=self.model_dir)

            weights = None
            
            if weights_path is None:
                weights = model.find_last()
            else:
                weights = weights_path

            model.load_weights(weights, by_name=True)

            self.testModel = model

        results = []
        for image in images:
            results.append(self.testModel.detect([image])[0])

        if verbose:
            r = results[0]
            visualize.display_instances(images[0], r['rois'], r['masks'], r['class_ids'], 
                                        ["",""], r['scores'],figsize=(10,10))

        print ('MaskRCNN Prediction complete after', time.time()-t0, 'seconds')

        return results
