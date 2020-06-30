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

    def segment_image_label(self, image_id):
        info  = self.image_info[image_id]
        image = info['image']
        bbox  = info['bbox']
        image = np.stack((image,)*3, -1)
        segmented_image = segment_images_label([image], [bbox], flat=True)
        return segmented_image



    def segment_dataset_label(self, flat=False, verbose=False):
        '''
        '''
        return segment_images_label(self.load_all_images(), self.bbox, flat, verbose)

    def segment_dataset_network(self, results, flat=False, verbose=False):
        '''
        '''
        return segment_images_network(self.load_all_images(), results, flat, verbose)


def segment_images_label(images, bbox, flat=False, verbose=False):
    '''
    '''

    return __segment_images(images, bbox, flat=flat, verbose=verbose)


def segment_images_network(images, results, flat=False, verbose=False):
    '''
    '''
    return __segment_images(images, result['bbox'], result['scores'], flat, verbose)


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
        isolated_images = []

        for r in rois:
            
            cut_image = image[r[0]:r[2],r[1]:r[3]]
            pad_cut_image = np.zeros((1,100,100,3),dtype=cut_image.dtype)
            befY = 50-(cut_image.shape[0] // 2)
            befX = 50-(cut_image.shape[1] // 2)
            pad_cut_image[0,befY:befY+cut_image.shape[0],befX:befX+cut_image.shape[1]] = cut_image

            from_left_to_right.append(r[1])
            
            isolated_images.append(pad_cut_image)
            
            if verbose:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(pad_cut_image[0])

        # sort scores back into the original order
        _, ordered_isolated_images = zip(*sorted(zip(from_left_to_right, isolated_images), key=lambda x: x[0]))

        # fixing weird singleton tuple issue
        ordered_isolated_images = [image[0] for image in ordered_isolated_images]

        if len(ordered_isolated_images) < 4:
            print(j)

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

    segmented_images -= .5


    return segmented_images



def normalize_labels(labels):

    l_min = labels.min()
    l_max = labels.max()

    labels -= l_min
    labels = labels / (l_max - l_min)
    labels -= .5

    return labels
