
import os, sys
from perception import maskr as m
from perception import angledataset as a
from ResultAnalyser import ResultAnalyzer
from keras.callbacks import History
from mrcnn import visualize
import matplotlib.pyplot as plt
import argparse

# getting access to the generator code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/generators/'))
from dataset import DatasetFromFile


class segmentation:
    def __init__(self):
        pass

    def visualizeDataset(self):
        train, val, test = a.AngleDataset.createSets()
        f, ax = plt.subplots()
        f.suptitle("Angle: " + str(test.image_info[0]['angles']), fontsize=16)
        ax.imshow(test.image_info[0]['mask'])
        ax.set_title("Original")
        plt.show()

    def trainMaskRCNN(self, epochs, dataset):
        maskrcnn = m.MaskR()
        path = dataset + "/"
        history = History()
        train_file = DatasetFromFile(path + "train_0.npz").load_from_file()
        val_file = DatasetFromFile(path + "val_0.npz").load_from_file()


        maskrcnn.train(train_file, val_file, history, epochs=epochs)
        return maskrcnn, history

    def predictOneMaskRCNN(self, ANY_INDEX, dataset):
        maskrcnn = m.MaskR()
        path = dataset + "/"
        test = DatasetFromFile(path + "test_0.npz").load_from_file()
        x_test = test.load_image(ANY_INDEX)
        maskrcnn_results = maskrcnn.predict([x_test], verbose=False)
        r = maskrcnn_results[0]
        visualize.display_instances(x_test, r['rois'], r['masks'], r['class_ids'],
                                    test.class_names, r['scores'])
        return 0


def main():
    print("-----LEVEL SEGMENTATION TRAINING PIPELINE-----")
    parser = argparse.ArgumentParser("level")
    parser.add_argument("dataset",
                        help="dataset can be any one of:  angle or area or curvature or direction or length or position_common_scale or position_non_aligned_scale or volume ",
                        type=str)
    parser.add_argument("path", help="dataset file path", type=str)
    parser.add_argument("epoch", help="val data count", type=int)
    parser.add_argument("historypath", help="local folder path to save history", type=str)
    parser.add_argument("gpu", help="select the gpu", type=int)

    args = parser.parse_args()
    dataset_path = ''
    if args.dataset == "angle":
        dataset_path = args.path + '/' + 'angle'
    elif args.dataset == "area":
        dataset_path = args.path + '/' + 'area'
    elif args.dataset == "curvature":
        dataset_path = args.path + '/' + 'curvature'
    elif args.dataset == "direction":
        dataset_path = args.path + '/' + 'direction'
    elif args.dataset == "length":
        dataset_path = args.path + '/' + 'length'
    elif args.dataset == "position_common_scale":
        dataset_path = args.path + '/' + 'position_common_scale'
    elif args.dataset == "position_non_aligned_scale":
        dataset_path = args.path + '/' + 'position_non_aligned_scale'
    elif args.dataset == "volume":
        dataset_path = args.path + '/' + 'volume'

    segmentation_pipeline = segmentation()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    result, history = segmentation_pipeline.trainMaskRCNN(args.epoch, dataset_path)
    # saving the history
    result_analyzer = ResultAnalyzer()
    res = result_analyzer.saveHistoryToLocalFile(args.historypath, history)
    if res == 0:
        print("saved history succesfully")
    else:
        print("saving history error")

    # get history from local and visualize
    hist = result_analyzer.loadHistoryFromLocalFile(args.historypath + "/history.p")
    result_analyzer = ResultAnalyzer()
    result_analyzer.plotLoss(int(args.epoch), hist)

    # load model and predict
    prediction = segmentation_pipeline.predictOneMaskRCNN(10, dataset_path)
    if prediction == 0:
        print("MaskRCNNModel prediction successful")
    else:
        print("error in prediction")


if __name__ == '__main__':
    main()
