from perception import maskr as m
from perception import angledataset as a
from ResultAnalyser import ResultAnalyzer
from keras.callbacks import History
import pickle
from sklearn.metrics import mean_squared_error
from mrcnn import visualize
import matplotlib
matplotlib.use('agg')


import matplotlib.pyplot as plt
import argparse

from dataset import DatasetFromFile

class segmentation:
    def __init__(self):
        pass

    def visualizeDataset(self):

        train, val, test = a.AngleDataset.createSets()
        f, ax = plt.subplots()
        f.suptitle("Angle: "+ str(test.image_info[0]['angles']), fontsize=16)
        ax.imshow(test.image_info[0]['mask'])
        ax.set_title("Original")
        plt.show()

    def trainMaskRCNN(self,epochs,dataset):
        maskrcnn = m.MaskR()
        path = dataset + "/"
        history = History()
        train_file = DatasetFromFile(path + "train_0.npz").load_from_file()
        val_file   = DatasetFromFile(path + "val_0.npz").load_from_file()
        maskrcnn.train(train_file, val_file,history, epochs=epochs)
        return maskrcnn, history

    def predictMaskRCNN(self,ANY_INDEX,test,maskrcnn,weight_path):
        x_test = test.load_image(ANY_INDEX)
        y_test = test.image_info[ANY_INDEX]['angles']
        maskrcnn_results = maskrcnn.predict([x_test], verbose=False)
        result_analyzer = ResultAnalyzer()
        load_model = result_analyzer.LoadModelfromLocal(weight_path)

        r = maskrcnn_results[0]
        visualize.display_instances(x_test, r['rois'], r['masks'], r['class_ids'],
                                    test.class_names, r['scores'])

        y_pred = load_model.predict([x_test], [maskrcnn_results])

        return mean_squared_error(y_test, y_pred[0])

    def saveHistoryToLocalFile(self,path,history):
        with open(path + "/history.p", 'wb') as file:
            pickle.dump(history.history, file, pickle.HIGHEST_PROTOCOL)
        return 0

    def loadHistoryFromLocalFile(self,path):
        with open(path, 'rb') as file:
            history = pickle.load(file)
        return history


    def testTrainedModel(self,dataset,weights_path):
        segmentation_pipeline = segmentation()
        result_analyzer=ResultAnalyzer()
        path = dataset + "/"
        test = DatasetFromFile(path + "test_0.npz").load_from_file()

        maskrcnn = result_analyzer.LoadModelfromLocal(weights_path)
        segmentation_pipeline.predictMaskRCNN(10, maskrcnn, test, weights_path)

def main():
    print("-----LEVEL TRAINING PIPELINE-----")
    parser = argparse.ArgumentParser("level")
    parser.add_argument("dataset", help="dataset can be any one of:  angle or area or curvature or direction or length or position_common_scale or position_non_aligned_scale or volume ", type=str)
    parser.add_argument("path", help="dataset file path", type=str)
    parser.add_argument("epoch", help="val data count", type=int)
    parser.add_argument("historypath", help="local folder path to save history", type=str)
    parser.add_argument("weightspath", help="local folder path to load weights", type=str)

    args = parser.parse_args()
    dataset_path=''
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
    result, history = segmentation_pipeline.trainMaskRCNN(args.epoch, dataset_path)
    #saving the history
    res = segmentation_pipeline.saveHistoryToLocalFile(args.historypath, history)
    if res==0:
        print("saved history succesfully")
    else:
        print("saving history error")

    #get history from local and visualize
    hist = segmentation_pipeline.loadHistoryFromLocalFile(args.historypath + "/history.p")
    result_analyzer = ResultAnalyzer()
    result_analyzer.plotAccuracy(hist)

    #load model and predict
    mse = segmentation_pipeline.testTrainedModel(dataset_path,args.weightspath)
    print("Mean square error from model" + str(mse))


if __name__ == '__main__':
    main()


