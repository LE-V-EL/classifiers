################################################################
# This file contains the pipeline for LE-V-EL platform.
#
#
#
################################################################

import os, sys
import numpy as np
import time

# gaining access to the generator code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/generators/'))
import figure5 as f5
from classifiers import classifiers


def generateDataset(dataset,train_target,val_target,test_target,flags):

    train_counter = 0
    val_counter = 0
    test_counter = 0
    NOISE=True

    # get global min and max
    global_min = np.inf
    global_max = -np.inf
    for N in range(train_target + val_target + test_target):
        if dataset == "angle":
            sparse, image, label, parameters = f5.angle(flags)
        elif dataset == "area":
            sparse, image, label, parameters = f5.area(flags)
        elif dataset == "curvature":
            sparse, image, label, parameters = f5.curvature(flags)
        elif dataset == "direction":
            sparse, image, label, parameters = f5.direction(flags)
        elif dataset == "length":
            sparse, image, label, parameters = f5.length(flags)
        elif dataset == "position_common_scale":
            sparse, image, label, parameters = f5.position_common_scale(flags)
        elif dataset == "position_non_aligned_scale":
            sparse, image, label, parameters = f5.position_non_aligned_scale(flags)
        elif dataset == "volume":
            sparse, image, label, parameters = f5.volume(flags)

        global_min = min(label, global_min)
        global_max = max(label, global_max)
    # end of global min max

    X_train = np.zeros((train_target, 100, 100), dtype=np.float32)
    y_train = np.zeros((train_target), dtype=np.float32)
    train_counter = 0

    X_val = np.zeros((val_target, 100, 100), dtype=np.float32)
    y_val = np.zeros((val_target), dtype=np.float32)
    val_counter = 0

    X_test = np.zeros((test_target, 100, 100), dtype=np.float32)
    y_test = np.zeros((test_target), dtype=np.float32)
    test_counter = 0

    t0 = time.time()

    min_label = np.inf
    max_label = -np.inf

    all_counter = 0
    while train_counter < train_target or val_counter < val_target or test_counter < test_target:
        all_counter += 1
        image = image.astype(np.float32)

        pot = np.random.choice(3)  # , p=([.6,.2,.2]))
        if label == global_min or label == global_max:
            pot = 0  # for sure training

        if pot == 0 and train_counter < train_target:
            # a training candidate
            if label in y_val or label in y_test:
                # no thank you
                continue

            # add noise?
            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))

            # safe to add to training
            X_train[train_counter] = image
            y_train[train_counter] = label
            train_counter += 1

        elif pot == 1 and val_counter < val_target:
            # a validation candidate
            if label in y_train or label in y_test:
                # no thank you
                continue

            # add noise?
            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))

            # safe to add to validation
            X_val[val_counter] = image
            y_val[val_counter] = label
            val_counter += 1

        elif pot == 2 and test_counter < test_target:
            # a test candidate
            if label in y_train or label in y_val:
                # no thank you
                continue

            # add noise?
            if NOISE:
                image += np.random.uniform(0, 0.05, (100, 100))

            # safe to add to test
            X_test[test_counter] = image
            y_test[test_counter] = label
            test_counter += 1

    print('Done', time.time()-t0, 'seconds (', all_counter, 'iterations)')
    # NORMALIZE DATA IN-PLACE (BUT SEPERATELY)
    X_min = X_train.min()
    X_max = X_train.max()
    y_min = y_train.min()
    y_max = y_train.max()

    # scale in place
    X_train -= X_min
    X_train /= (X_max - X_min)
    y_train -= y_min
    y_train /= (y_max - y_min)

    X_val -= X_min
    X_val /= (X_max - X_min)
    y_val -= y_min
    y_val /= (y_max - y_min)

    X_test -= X_min
    X_test /= (X_max - X_min)
    y_test -= y_min
    y_test /= (y_max - y_min)

    # normalize to -.5 .. .5
    X_train -= .5
    X_val -= .5
    X_test -= .5

    return X_train, X_val, X_test, y_train, y_val, y_test

def consumeDatsetfromFile(dataset,path):
    if dataset == "angle":
        path = path + '/' + 'angle'
    elif dataset == "area":
        path = path + '/' + 'area'
    elif dataset == "curvature":
        path = path + '/' + 'curvature'
    elif dataset == "direction":
        path = path + '/' + 'direction'
    elif dataset == "length":
        path = path + '/' + 'length'
    elif dataset == "position_common_scale":
        path = path + '/' + 'position_common_scale'
    elif dataset == "position_non_aligned_scale":
        path = path + '/' + 'position_non_aligned_scale'
    elif dataset == "volume":
        path = path + '/' + 'volume'

    train = np.load(path + '/' + 'train_0.npz')
    test = np.load(path + '/' + 'test_0.npz')
    val = np.load(path + '/' + 'val_0.npz')

    return train,test,val

import argparse



def main():
    print("-----LEVEL TRAINING PIPELINE-----")
    parser = argparse.ArgumentParser("level")
    parser.add_argument("dataset", help="dataset can be any one of:  angle or area or curvature or direction or length or position_common_scale or position_non_aligned_scale or volume ", type=str)
    parser.add_argument("path", help=" Path to store the model", type=str)
    parser.add_argument("classifier", help="classifier can be VGG19 or DenseNet121 or ALEXNET", type=str)
    parser.add_argument("train", help="training data count", type=int)
    parser.add_argument("test", help="test data count", type=int)
    parser.add_argument("val", help="val data count", type=int)
    parser.add_argument("epoch", help="val data count", type=int)
    parser.add_argument("batchsize", help="val data count", type=int)
    parser.add_argument("optimizer", help="optimizer can be:  ", type=str)
    parser.add_argument("error", help="error can be:  ", type=str)
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = generateDataset(str(args.dataset), args.train, args.val, args.test)
    print("dataset prepared")
    classifer = classifiers()
    if args.classifier == "ALEXNET":
        model,X_train_3D,X_val_3D,X_test_3D = classifer.createScratchClassifiers(args.classifier, X_train, X_val, X_test,args.optimizer,args.loss)
        classifer.trainClassifer(model, args.classifer, X_train_3D, y_train,X_val_3D,y_val,args.epoch,args.batchsize,args.path)
    else:
       feature_generator,X_train_3D,X_val_3D,X_test_3D =  classifer.createKerasClassifiers(args.classifier,X_train,X_val,X_test)
       model = classifer.normaliseClassifiers(feature_generator,args.optimizer,args.loss)



if __name__ == '__main__':
    main()
