import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #1: filter out info logs, 2: filter out warning logs, 3: filter out error logs
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

plt.rcParams.update({'font.size': 20})
plt.rcParams["figure.dpi"] = 200

#Fix for the Tensorflow error: Failed to get convolution algorithm.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('')


#Split the complete slice layer into not overlapping patches of 128x128 pixels,
#which will be used for classification
def split_image(source_image, window_size, window_overlap, source_dim):

    patch_data = []

    #Check if there is a 50% overlap between patches or no overlap at all.
    #Required for the patch behavior on the right and lower sides
    if window_overlap == window_size:
       last_step = 0

    else:
       last_step = window_overlap

    for ypos in range(0, (source_dim - last_step), window_overlap):
            for xpos in range(0, (source_dim - last_step), window_overlap):
                
                #Initialize the arrays to contain the patch
                sliding_window = np.zeros(shape=(window_size, window_size, 3),dtype=np.uint8)
                image_patch = np.zeros(shape=(window_size, window_size, 3),dtype=np.uint8)

                #Copy each image data channel from the source image to the patch
                for channel in range(0, 3):
                    sliding_window = source_image[ypos : (ypos + window_size), xpos : (xpos + window_size), channel]
                    image_patch[:, :, channel] = sliding_window

                #And store the patch in a list
                patch_data.append(image_patch)
                
    return np.asarray(patch_data)


#This function will create an overlay image. Each Patch will have either the color
#White (Powder), blue (Object) or red (Error)
#Inputs:
#   predictions_logits: Predicted logits for each patch from the CNN (0: Pulver, 1: Bauteil, 2: Fehler)
#   predictions_logits has 36 elements. (patch size 128, source dim 768x768, no overlap -> 768/128=6 -> 6x6=36 Patches)
def stitch_image(predictions_logits, window_size, window_overlap, source_dim):

    #Initialize an empty array with the shape of the final image (the extracted powderbed)
    image = np.zeros(shape=(source_dim, source_dim, 3), dtype=np.uint8)

    #Check if there is a 50% overlap between patches or no overlap at all.
    if window_overlap == window_size:
       last_step = 0

    else:
       last_step = window_overlap

    #Counter to determine patch number/patch location
    i = 0

    for ypos in range(0, (source_dim - last_step), window_overlap):
            for xpos in range(0, (source_dim - last_step), window_overlap):

                #initialize a patch
                color = np.zeros(shape=(window_size, window_size, 3), dtype=np.uint8)

                #Set the patch color according to the CNN Label
                if predictions_logits[i] == 0:
                   color[...,0:3]=[255,255,255] #white for powder
                elif predictions_logits[i] == 1:
                   color[...,0:3]=[0,0,255] #blue for object
                else:
                   color[...,0:3]=[255,0,0] #red for error

                #Fill the empty powderbed array from above with the patches and their colors
                for channel in range(0, 3):
                    image[ypos : (ypos + window_size), xpos : (xpos + window_size), channel] = color[:,:,channel]

                #increase patch number by 1 (move to the next patch)
                i = i + 1

    return image


#Function to load and decode a PNG image
def load_image(img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3, dtype=np.uint8)
    
    return np.asarray(image)


#Normalize the image patches before making predictions on them
def normalize(image_patches):
    
    image_patches = tf.image.convert_image_dtype(image_patches, tf.float32)
    image_patches = tf.image.per_image_standardization(image_patches)

    return image_patches


def main():
    #########################################################################
    #Function to visualize an entire layer by the CNN classifier
    #########################################################################

    #Path to the CNN Model
    os.chdir('..')
    model_path = 'model\\'
    #Path to the folder containing the entire layer images
    layers_folder = 'Test_Data\\Layers\\'

    patch_size = 128
    patch_overlap = 128 #if patch_size == patch_overlap -> No overlap between two patches
    source_image_dim = 768 #Powderbed dimension
    
    #Load the cnn model
    cnn_model = tf.keras.models.load_model(model_path)

    #Get the list of all image filenames and their true slice labels.
    #Note: Slice Label is binary and applies to the whole Slice Layer. 0 -> Good Powderbed. 1 -> There is an error in the Layer
    project_images = [layers_folder + s for s in os.listdir(layers_folder)]

    print('Note:\nClose the Layer Image to show the next one.')

    for i in range(len(project_images)):

        fig, ax = plt.subplots()
        plt.title('Powder: White | Object: Blue | Error: Red')
        #Load the image file (specify the image path to a slice)
        image_data = load_image(project_images[i])
        #split the (extracted) powderbed image into not overlapping patches and stack them into a list
        patch_array = split_image(image_data, patch_size, patch_overlap, source_image_dim)
        #Normalize the patches  for prediction (subtract mean and divide by standard deviation)
        patches_normalized = normalize(patch_array)
        #predict all patches from one powderbed (Layer)
        y_predicted_raw = cnn_model.predict(patches_normalized, verbose=0)
        #Get the CNN Logits with argmax for each patch
        y_predicted_logits = np.argmax(y_predicted_raw, axis=-1)

        #Create the chessboard like image with each patch containing the color of their label.
        stitched_image = stitch_image(y_predicted_logits, patch_size, patch_overlap, source_image_dim)
        #overlay the original image with the color map to show how each patch was classified
        overlayed_images = cv2.addWeighted(image_data, 0.8, stitched_image, 0.2, 0)

        ax.imshow(overlayed_images, interpolation='nearest')
        ax.xaxis.tick_top()
        fig.canvas.draw()
        ax.set_xticks(np.arange(patch_size/2, source_image_dim, patch_size))
        ax.set_xticklabels(np.arange(1,7,1))
        ax.set_yticks(np.arange(patch_size/2, source_image_dim, patch_size))
        ax.set_yticklabels(np.arange(1,7,1))

        plt.show()




if __name__ == '__main__':
    main()
