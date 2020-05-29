from perception import maskr as m
from perception import angledataset as a
from ResultAnalyser import ResultAnalyzer
import matplotlib.pyplot as plt

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

    def trainMaskRCNN(self,train,val,epochs):
        maskrcnn = m.MaskR()
        #change to npy consume
        train, val, test = a.AngleDataset.createSets()
        maskrcnn.train(train, val, epochs=epochs)
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

segmentation_pipeline = segmentation()
result = segmentation_pipeline.trainMaskRCNN(60000,20000,1000)

#save model
result_analyzer = ResultAnalyzer()
model_path='/models'
result_analyzer.saveModels("segementationMaskRCNN",result,model_path,1000)

#load model and predict
weights_path =""
maskrcnn = result_analyzer.LoadModelfromLocal(model_path,weights_path)
train, val, test = a.AngleDataset.createSets()
segmentation_pipeline.predictMaskRCNN(10,maskrcnn,test,model_path,weights_path)