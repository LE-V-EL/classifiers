import os, sys, time

from sklearn.metrics import mean_squared_error

import numpy as np

import perception.maskr       as m
import perception.vgg19bridge as v
import perception.dataset     as d

class Classifier:



    def __init__(self, storage_dir=None):

        if storage_dir is not None:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = os.path.join(os.path.dirname(__file__))



    def initialize(self, model_name=None):
        '''
        '''
        if model_name is not None:
            self.model_dir  = os.path.join(self.storage_dir, model_name)
            self.model_name = model_name

        else:
            self.model_dir = os.path.join(self.storage_dir, "model")

            magic_number = 0
            if os.path.exists(self.model_dir + str(magic_number)):
                # we need to increase the magicnumber until we find a good one
                while os.path.exists(self.model_dir + str(magic_number)):
                    magic_number += 1

            self.model_dir  = self.model_dir + str(magic_number)
            self.model_name = "model" + str(magic_number)

            os.mkdir(self.model_dir)

            print ('Storing in ', self.model_dir)


        maskrcnn_dir = os.path.join(self.model_dir, "maskrcnn")

        if not os.path.exists(maskrcnn_dir):
            os.mkdir(maskrcnn_dir)

        self.maskrcnn  = m.MaskR(maskrcnn_dir)


        vgg19_dir = os.path.join(self.model_dir, "vgg19")

        if not os.path.exists(vgg19_dir):
            os.mkdir(vgg19_dir)

        self.vgg19 = v.VGG19Bridge(vgg19_dir)



    def train(self, maskrcnn_epochs, vgg19_epochs):
        '''
        '''
        self.initialize()

        m_train   = d.Dataset(os.path.join(self.storage_dir, "dataset", "train_0.npz"))
        m_val     = d.Dataset(os.path.join(self.storage_dir, "dataset", "val_0.npz"))

        history = []
        if maskrcnn_epochs > 0:
            m_history = self.maskrcnn.train(m_train, m_val, epochs=maskrcnn_epochs)
            history.append(m_history.history)

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
            history.append(v_history.history)

        return history



    def predict(self, image):

        segmentation_pred = self.maskrcnn.predict([image])

        segmented_image = d.segment_image_network(image, segmentation_pred[0])

        prediction = self.vgg19.predict(segmented_image)

        return prediction



    def test(self, model_name=None):
        '''
        '''
        # this is the case when you havent been training on this object
        if model_name is not None:
            self.initialize(model_name)

        m_test = d.Dataset(os.path.join(self.storage_dir, "dataset", "test_0.npz"))

        labels = m_test.labels

        results = []
        bad_result_ids = []
        
        t0 = time.time()

        for image_id in range(len(m_test.image_info)):

            image = m_test.load_image(image_id)

            prediction = self.predict(image)

            # a result is bad when it has too few stimuli detectted
            if len(prediction[0]) < labels.shape[1]:
                bad_result_ids.append(image_id)

            results.extend(prediction)

        print ('Predictions completed after', time.time()-t0)

        l_max = labels.max()
        l_min = labels.min()

        labels = labels.tolist()

        bad_results = []
        bad_labels  = []

        if len(bad_result_ids) > 0:
            # poping the bad results from the back so that we dont 
            # pop the wrong ones
            bad_result_ids.sort(reverse=True)
            for image_id in bad_result_ids:
                bad_labels.append(labels.pop(image_id))
                bad_results.append(results.pop(image_id))

        labels      = np.array(labels)
        bad_labels  = np.array(bad_labels)

        # not denormalizing the bad results since their length is not consistent
        # and that breaks the denormalization code.
        results     = d.denormalize_results(results, l_max, l_min)

        msqr_error  = mean_squared_error(labels, results)

        file = os.path.join(self.model_dir, "results.npz")

        np.savez(file,
            labels=labels,
            results=results,
            bad_labels=bad_labels,
            bad_results=bad_results,
            bad_result_ids=np.array(bad_result_ids),
            msqr_error=msqr_error
        )


