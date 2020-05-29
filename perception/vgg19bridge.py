import os, sys, time

import keras
from keras.models import load_model

import numpy as np

class VGG19Bridge:

    def __init__(self, modelpath='tensorflow.js/models/VGG19_angle_full_variation/01_noise.h5'):
        '''
        '''
        t0 = time.time()
        self.model = load_model(os.path.join(os.path.dirname(__file__), modelpath))
        print ('Setup complete after', time.time()-t0)


    def predict(self, images, results, verbose=False):
        '''
        Predicts a maskr-cnn results dict using VGG19.
        '''
        t0 = time.time()

        all_preds = []

        for j,image in enumerate(images):

            result = results[j]

            scores = result[0]['scores']
            rois = result[0]['rois'] # rois are y1, x1, y2, x2 

            # sort by score
            scores2, rois2 = zip(*sorted(zip(scores,rois)))
            scores = scores2[-4:]
            rois = rois2[-4:] # top 4

            vgg_scores = []
            from_left_to_right = []
            isolated_images = []

            for r in rois:
                
                cut_image = image[r[0]-10:r[2]+10,r[1]-10:r[3]+10]
                pad_cut_image = np.zeros((1,100,100,3),dtype=cut_image.dtype)
                befY = 50-(cut_image.shape[0] // 2)
                befX = 50-(cut_image.shape[1] // 2)
                pad_cut_image[0,befY:befY+cut_image.shape[0],befX:befX+cut_image.shape[1]] = cut_image

                pad_cut_image_norm = pad_cut_image / 255.
                pad_cut_image_norm += np.random.uniform(0, 0.05,(1,100,100,3))

                X_min = pad_cut_image_norm.min()
                X_max = pad_cut_image_norm.max()

                # scale in place
                pad_cut_image_norm -= X_min
                pad_cut_image_norm /= (X_max - X_min)

                pad_cut_image_norm -= .5
                
                # predict
                vgg_scores.append(self.model.predict(pad_cut_image_norm)[0])
                
                from_left_to_right.append(r[1])
                
                isolated_images.append(pad_cut_image_norm)
                
                if verbose:
                    plt.figure()
                    imshow(pad_cut_image[0])
                
            # sort scores back into the original order
            from_left_to_right,vgg_scores = zip(*sorted(zip(from_left_to_right,vgg_scores)))

            y_image_pred = []
            for v in vgg_scores:
                y_image_pred.append(v[0]*90)

            all_preds.append(y_image_pred)

        print('Prediction complete after', time.time()-t0)

        return all_preds
