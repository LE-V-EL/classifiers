import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model

class ResultAnalyzer:
    def __init__(self):
        pass

    # Plot the loss from each batch
    def plotLoss(self, epoch,Losses):
        plt.figure(figsize=(10, 8))
        plt.plot(Losses, label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/loss_epoch_%d.png' % epoch)

    def plotAccuracy(self,history):
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def saveModels(self,name, model, model_path,epoch):
        model_json = model.to_json()
        with open(model_path + '/' + name + "classifer" +epoch +".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_path + '/' + name + "classifer" +epoch +".h5")
        print("Model and Weights saved")

    def visualizeModelGraph(self,model,filename):
        model.summary()
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

    def LoadModelfromLocal(self,model_weights):
        loaded_model = load_model(model_weights)
        print("model loaded successfully")
        return loaded_model

