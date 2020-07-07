import os, sys, time

import numpy as np

# we need access to the MaskR-CNN code
sys.path.append('../external/mask_rcnn/')
from mrcnn import utils
from mrcnn import visualize


class Dataset(utils.Dataset):

    def __init__(self, file):
        
        super().__init__()

        self.file = file
        self.metadata = None
        self.max = 0
        self.load_from_file()


    @staticmethod
    def load_from_file_raw(file):
        data = np.load(file)

        # pulling all data out of npz file
        loaded_data = dict(map(lambda component: ( component, data[component] ), data.keys()))

        return loaded_data


    def load_from_file(self):
        '''
        '''

        loaded_data = Dataset.load_from_file_raw(self.file)

        self.max = np.max(loaded_data["image"])

        #normalize data 0 - 1
        loaded_data["image"] /= self.max

        self.metadata = dict(loaded_data.pop('metadata'))

        # saving these in a less annoying format also
        self.labels = loaded_data["label"]
        self.bbox   = loaded_data["bbox"]

        SETNAME = 'stimuli'
        
        self.add_class(SETNAME, 1, self.metadata['data_class'])

        for i in range(len(loaded_data['label'])):

            # pulling out all the info for one image
            image_data = dict( map(lambda component: ( component, loaded_data[component][i] ), loaded_data.keys()))

            self.add_image(SETNAME,
                image_id   = i,
                path       = None,
                **image_data)

        self.prepare()

        return self



    def load_image(self, image_id):
        '''
        '''
        info = self.image_info[image_id]

        image = info['image']
        
        loaded_img_3D = np.stack((image,)*3, -1)
        
        return (loaded_img_3D*255).astype(np.uint8)



    def load_mask(self, image_id):
        '''
        '''
        mask = self.image_info[image_id]['mask']

        # it is always class 1 but for the amount of stimuli
        return mask, np.ones(mask.shape[2], dtype='uint8')



    def show(self, howmany=1):
        '''
        '''
        image_ids = np.random.choice(self.image_ids, howmany)
        
        for image_id in image_ids:
            
            image = self.load_image(image_id)
            
            mask, class_ids = self.load_mask(image_id)

            visualize.display_top_masks(image, mask, class_ids, self.class_names)



    def show_bbox(self, image_id):
        '''
        '''
        image = self.load_image(image_id)

        mask, class_ids = self.load_mask(image_id)

        bbox = self.image_info[image_id]['bbox']

        visualize.display_instances(image, bbox, mask, class_ids,
                            self.class_names, figsize=(8, 8))


    def load_all_images(self):
        '''
        '''
        images = []

        for image_id in range(self.num_images):

            images.append(self.load_image(image_id))

        return images



    def segment_image_label(self, image_id, verbose=False):
        ''''''
        image = self.load_image(image_id)
        info  = self.image_info[image_id]
        bbox  = info['bbox']
        segmented_image = segment_image_label(image, bbox, flat=True, verbose=verbose)
        return segmented_image



    def segment_image_network(self, image_id, result, verbose=False):
        '''
        '''
        image = self.load_image(image_id)
        segmented_image = segment_image_network(image, result[0], flat=True, verbose=verbose)
        return segmented_image


def segment_image_label(image, bbox, flat=False, verbose=False):
    '''
    '''
    return __segment_images([image], [bbox], flat=flat, verbose=verbose)


def segment_image_network(image, result, flat=False, verbose=False):
    '''
    '''
    return __segment_images([image], [result['rois']], [result['scores']], flat, verbose)


def __segment_images(images, bbox, all_scores=None, flat=False, verbose=False):
    '''
    flat: instead of a list of lists with segments grouped by origial image, it returns a flat list of the segments
    '''


    if all_scores is None:
        #no sort needed
        all_scores = [[1,1,1,1]] * len(bbox)

    segmented_images = []

    for j,image in enumerate(images):

        scores = all_scores[j]
        rois   = bbox[j] # rois are y1, x1, y2, x2 

        # sort by score
        scores2, rois2 = zip(*sorted(zip(scores,rois), key=lambda x: x[0]))
        scores = scores2[-4:]
        rois = rois2[-4:] # top 4

        from_left_to_right = []
        from_top_to_bottom = []
        isolated_images = []

        order_index = 0


        for r in rois:

            # detecting image boundaries to prevent negative index 
            # among other issues
            y1 = r[0] - 10
            if y1 < 0:
                y1 = 0

            y2 = r[2] + 10
            if y2 > image.shape[0]:
                y2 = image.shape[0]

            x1 = r[1] - 10
            if x1 < 0:
                x1 = 0

            x2 = r[3] + 10
            if x2 > image.shape[1]:
                x2 = image.shape[1]

            cut_image = image[y1:y2,x1:x2]
            pad_cut_image = np.zeros((1,100,100,3),dtype=cut_image.dtype)
            befY = 50-(cut_image.shape[0] // 2)
            befX = 50-(cut_image.shape[1] // 2)
            pad_cut_image[0,befY:befY+cut_image.shape[0],befX:befX+cut_image.shape[1]] = cut_image

            pad_cut_image = pad_cut_image / 255.

            from_left_to_right.append(r[1])

            from_top_to_bottom.append(r[0])
            
            isolated_images.append(pad_cut_image)
            
            if verbose:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(pad_cut_image[0])

        order = from_left_to_right
        # this is the case where the stimuli are nearly the same
        # x value so we instead order them from top to bottom
        # (postion_common_scale)
        if max(from_left_to_right) - min(from_left_to_right) < 10:
            order = from_top_to_bottom

        ordered_isolated_images = []
        if len(isolated_images) > 0:
            # sort scores back into the original order
            _, ordered_isolated_images = zip(*sorted(zip(order, isolated_images), key=lambda x: x[0]))

            # fixing weird singleton tuple issue
            ordered_isolated_images = [image[0] for image in ordered_isolated_images]

        if flat:
            segmented_images.extend(ordered_isolated_images)
        else:
            segmented_images.append(ordered_isolated_images)

    segmented_images = np.asarray(segmented_images)

    #segmented_images = segmented_images / 255.
    segmented_images += np.random.uniform(0, 0.05, segmented_images.shape)

    X_min = segmented_images.min()
    X_max = segmented_images.max()

    # scale in place
    segmented_images -= X_min
    segmented_images /= (X_max - X_min)

    # values to -.5 - .5
    segmented_images -= .5

    return segmented_images



def normalize_labels(labels):

    l_min = labels.min()
    l_max = labels.max()

    labels -= l_min
    labels = labels / (l_max - l_min)
    labels -= .5

    return labels

def denormalize_results(results, l_max, l_min):

    denorm_results = np.array(results)
    denorm_results += .5
    denorm_results *= (l_max - l_min)
    denorm_results += l_min

    return denorm_results
