from perception import maskr as m
from perception import angledataset as a
from ResultAnalyser import ResultAnalyzer
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
        train_file = DatasetFromFile(path + "train_0.npz").load_from_file()
        val_file   = DatasetFromFile(path + "val_0.npz").load_from_file()
        maskrcnn.train(train_file, val_file, epochs=epochs)
        return maskrcnn

    def predictMaskRCNN(self,ANY_INDEX,test,maskrcnn,model_path,weight_path):
        x_test = test.load_image(ANY_INDEX)
        y_test = test.image_info[ANY_INDEX]['angles']
        maskrcnn_results = maskrcnn.predict([x_test], verbose=False)
        result_analyzer = ResultAnalyzer()
        load_model = result_analyzer.LoadModelfromLocal(model_path,weight_path)
        from mrcnn import visualize
        r = maskrcnn_results[0]  # we only have one result since we only used one image to test
        visualize.display_instances(x_test, r['rois'], r['masks'], r['class_ids'],
                                    test.class_names, r['scores'])

        y_pred = load_model.predict([x_test], [maskrcnn_results])
        from sklearn.metrics import mean_squared_error
        mean_squared_error(y_test, y_pred[0])

    def testTrainedModel(self,dataset,weights_path):
        segmentation_pipeline = segmentation()
        path = dataset + "/"
        test = DatasetFromFile(path + "test_0.npz").load_from_file()
        maskrcnn = result_analyzer.LoadModelfromLocal(model_path, weights_path)
        segmentation_pipeline.predictMaskRCNN(10, maskrcnn, test, model_path, weights_path)

def main():
    print("-----LEVEL TRAINING PIPELINE-----")
    parser = argparse.ArgumentParser("level")
    parser.add_argument("dataset", help="dataset can be any one of:  angle or area or curvature or direction or length or position_common_scale or position_non_aligned_scale or volume ", type=str)
    parser.add_argument("path", help=" Path to store the model", type=str)
    parser.add_argument("epoch", help="val data count", type=int)
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
    result = segmentation_pipeline.trainMaskRCNN(args.epoch, dataset_path)


if __name__ == '__main__':
    main()







#save model
result_analyzer = ResultAnalyzer()
model_path='/models'
#result_analyzer.saveModels("segementationMaskRCNN",result,model_path,1000)

#load model and predict
