import os, sys

import perception.maskr       as m
import perception.vgg19bridge as v
import perception.dataset     as d

class Classifier:

    def __init__(self, storage_dir=None):

        if storage_dir is not None:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = os.path.join(os.path.dirname(__file__))

        if not os.path.exists(self.storage_dir):
            os.mkdir(self.storage_dir)

        self.model_dir = os.path.join(self.storage_dir, "model")

        magic_number = 0
        if os.path.exists(self.model_dir+str(magic_number)):
            # we need to increase the magicnumber until we find a good one
            while os.path.exists(self.model_dir+str(magic_number)):
                magic_number += 1

        self.model_dir = self.model_dir + str(magic_number)
        os.mkdir(self.model_dir)


        self.maskrcnn  = m.MaskR(os.path.join(self.model_dir, "maskrcnn"))
        self.vgg19     = v.VGG19Bridge(os.path.join(self.model_dir, "vgg19"))


    def train(self, maskrcnn_epochs, vgg19_epochs):

        m_train   = d.Dataset(os.path.join(self.storage_dir, "dataset", "train_0.npz"))
        m_val     = d.Dataset(os.path.join(self.storage_dir, "dataset", "val_0.npz"))

        if maskrcnn_epochs > 0:
            m_history = self.maskrcnn.train(m_train, m_val, epochs=maskrcnn_epochs)

        # with the segmenter trained, we now segment the images by the cannonical bounding boxes
        # and remove the grouping to make  a dataset for the rest of the training that isnt 
        # concerned with the original image.

        v_train_x = []

        for image_id in range(int(len(m_train.labels) / 4)):

            segmented_image = m_train.segment_image_label(image_id)
            v_train_x.extend(segmented_image)

        v_train_x = [v_train_x]

        v_train_y = m_train.labels

        del m_train

        v_train_y = d.normalize_labels(v_train_y.flatten())
        v_train_y = v_train_y[0:int(len(v_train_y)/4)]

        v_val_x = []
        for image_id in range(int(len(m_val.labels) / 4)):

            segmented_image = m_val.segment_image_label(image_id)
            v_val_x.extend(segmented_image)

        v_val_x = [v_val_x]

        v_val_y = m_val.labels

        del m_val

        v_val_y   = d.normalize_labels(v_val_y.flatten())
        v_val_y   = v_val_y[0:int(len(v_val_y)/4)]

        if vgg19_epochs > 0:
            v_history = self.vgg19.train(v_train_x, v_train_y, v_val_x, v_val_y, epochs=vgg19_epochs)

        return m_history, v_history
