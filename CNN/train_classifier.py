import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #1: filter out info logs, 2: filter out warning logs, 3: filter out error logs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from modules.pipeline import batch_dataset
from modules.utils import plot_loss_and_accuracy, list_shuffle, plot_cm
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D

plt.rcParams.update({'font.size': 16})

#Fix for the Tensorflow error: Failed to get convolution algorithm.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('')

#The CNN architecture is defined here
def classifier_model():

    inputs  = tf.keras.Input(shape=(128, 128, 3), name='image')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.10)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.20)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.30)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.30)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(3, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def main():
    #Classes to predict
    multiclass_classes = ['Powder', 'Object', 'Error']
    #Directory to the dataset containing the patches. 
    #Note: Specify the path to the folder containing the patches dataset and the .csv files.
    #'Training_dataset_patches.csv', 'Training_Data', 'Validation_dataset_patches.csv', 'Validation_Data'
    dataset_directory = 'C:\\Users\\adeli\\OneDrive\\FH\\TH_Stelle\\DGaO\\Datensatz_03.12.2020\\'
    #Path to store the trained model
    model_path = 'model\\'

    #Training parameters
    BATCH_SIZE = 64
    EPOCHS = 150
    VERBOSE = 1 #Limit Tensorflow output while training

    #Pre training cleanup (just to be sure)
    tf.keras.backend.clear_session()

    #Load training dataset
    train_dataframe = pd.read_csv(dataset_directory + 'Training_dataset_patches.csv', sep=';')

    #Create lists containing the training image paths and labels
    X_train = list(dataset_directory + train_dataframe['Image'])
    y_train = list(train_dataframe['Label'])

    #shuffle the training lists row wise (just to be sure it is shuffled) 
    X_train, y_train = list_shuffle(X_train, y_train, seed1=42, seed2=36, seed3=9324)

    #Load validation dataset and create lists
    val_dataframe = pd.read_csv(dataset_directory + 'Validation_dataset_patches.csv', sep=';')
    X_val = list(dataset_directory + val_dataframe['Image'])
    y_val = list(val_dataframe['Label'])

    #shuffle the validation lists row wise (just to be sure it is shuffled) 
    X_val, y_val = list_shuffle(X_val, y_val, seed1=22, seed2=423)

    #Convert lists to tensorflow dataset
    tf_train_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X_train), tf.data.Dataset.from_tensor_slices(y_train)))
    tf_val_dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(X_val), tf.data.Dataset.from_tensor_slices(y_val)))

    #Calculate number of steps per epoch
    train_steps = len(X_train)/BATCH_SIZE
    val_steps = len(X_val)/BATCH_SIZE
    
    #Call model function and compile
    model = classifier_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.summary()

    #Calculate the class weight from the training dataset because of its inbalance
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i : class_weights[i] for i in range(0, len(multiclass_classes))}

    #Early stopping configuration to stop the training after 20 Epochs and restore the best weights to decrease overfitting
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', patience=20, restore_best_weights=True)

    #Train the model
    history = model.fit(batch_dataset(tf_train_dataset, BATCH_SIZE, shuffle=True, predict=False), 
                        epochs=EPOCHS, 
                        verbose=VERBOSE, 
                        validation_data=batch_dataset(tf_val_dataset, BATCH_SIZE, shuffle=True, predict=True), 
                        class_weight=class_weights, 
                        steps_per_epoch=train_steps, 
                        validation_steps=val_steps, 
                        validation_freq=1,
                        use_multiprocessing=False,
                        callbacks=[callback])

    #Save the trained model and a picture of the architecture to disk
    model.save(model_path, save_format='tf')

    #Store the loss and accuracy values for each epoch
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    number_of_epochs_trained = len(loss)

    #Predict with the trained model and generate the confusion matrix on the validation dataset
    #Note: Validation Data is not shuffled for prediction to keep the order. Necessary for the confusion matrix
    y_predicted_raw = model.predict(batch_dataset(tf.data.Dataset.from_tensor_slices(X_val), BATCH_SIZE, shuffle=False, predict=True), verbose=0)
    #Get the logits from the class probabilities for each image
    y_predicted_logits = np.argmax(y_predicted_raw, axis=-1)
    
    #Plot the accuracy and loss curves
    plot_loss_and_accuracy(acc, val_acc, loss, val_loss, number_of_epochs_trained)

    #Plot confusion matrix
    plot_cm(y_val, y_predicted_logits, multiclass_classes)

    #Print the classification report
    print(classification_report(y_val, y_predicted_logits, target_names=multiclass_classes, digits=6))

    plt.show()





if __name__ == "__main__":
    main()