import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #1: filter out info logs, 2: filter out warning logs, 3: filter out error logs
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from modules.pipeline import batch_dataset
from modules.utils import plot_cm, plot_img_attributions

plt.rcParams.update({'font.size': 16})

#Fix for the Tensorflow error: Failed to get convolution algorithm.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('')

def main():
    #Classes to predict
    multiclass_classes = ['Powder', 'Object', 'Error']

    #Directory to the dataset containing the patches. 
    os.chdir('..')
    dataset_directory = 'Test_Data\\Patches\\'
    #Path to the trained model
    model_path = 'model\\'

    #Load pretrained model
    model = tf.keras.models.load_model(model_path)

    test_data = os.listdir(dataset_directory)
    print(test_data)

    test_data = [dataset_directory + s for s in test_data]
    print(test_data)

    #Create lists containing the training image paths and labels
    X_test = list(dataset_directory + test_dataframe['Image'])
    y_test = list(test_dataframe['Label'])

    #Predict with the trained model and generate the confusion matrix on the validation dataset
    #Note: Test Data is not shuffled for prediction to keep the order. Necessary for the confusion matrix
    y_predicted_raw = model.predict(batch_dataset(tf.data.Dataset.from_tensor_slices(X_test), 64, shuffle=False, predict=True), verbose=0)
    #Get the logits from the class probabilities for each image
    y_predicted_logits = np.argmax(y_predicted_raw, axis=-1)

    #Print classification report
    print(classification_report(y_test, y_predicted_logits, target_names=multiclass_classes, digits=6))

    print('Note: \nClose the Confusion Matrix window to show the heatmaps of the wrongly classified patches.')
    #Plot confusion matrix
    plot_cm(y_test, y_predicted_logits, multiclass_classes)

    plt.show()

    #This part will show the wrongly classified patches and the integrated gradients as an overlay
    #Close the Matplot window to show the next one.
    compare_arrays = (y_test == y_predicted_logits)
    indices = np.where(compare_arrays == False)[0]

    for i in range(0, len(indices)):

        x = tf.io.read_file(X_test[indices[i]])
        x = tf.image.decode_png(x, channels=3)

        print('True Label: ', multiclass_classes[int(y_test[indices[i]])])
        print('Predicted Label: ', multiclass_classes[y_predicted_logits[indices[i]]])

        baseline = tf.zeros(shape=(128, 128,3))

        target_class = int(y_test[indices[i]])


        _ = plot_img_attributions(model=model, image=x,
                                  baseline=baseline,
                                  target_class_idx=target_class,
                                  m_steps=500,
                                  cmap=plt.cm.inferno,
                                  overlay_alpha=0.3)

        plt.show()

if __name__ == "__main__":
    main()