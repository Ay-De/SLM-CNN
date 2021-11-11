import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import glob
import pandas as pd
import os, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #1: filter out info logs and 2 and 3, 2: filter out warning logs and 3, 3: filter out only error logs
import tensorflow as tf
import ctypes

#######################################################################################################
# Some global variables (i know, big nono)
#######################################################################################################
#Location of the dataset of the extracted powderbed
os.chdir('..')
dataset_directory = 'dataset\\'
#Location of the target directory of the processed dataset
dataset_patches = dataset_directory + 'Patches\\'
#Size of patches to extract in pixels
patch_size = 128
#Step size, which each patch makes. 0.5 = 50% overlap between two patches. 1 = no overlap (equals Patch size == Step size).
#Note: Overlap is only applied to training dataset. Validation and Test datasets are split without any overlapping patches
step_size = 0.5
#Dimensions in X, Y of the source image
source_image_dim = 768

#Lists to store the extracted image patches and labels
patches_raw = []
labels_raw = []

#Variable to store the indices of patches containing an error
patch_label_raw_idx = []
#Total number of labels containing an error (len(patch_label_raw_idx))
patch_n_max = 0
#Number of the currently processed patch
patch_n = 0

#Labels:
#No Error (Placeholder): 0
#Powder (good): 1
#Object: 2
#Error: 3
labels_list = ['(Ignore this Button)', 'Powder (good)', 'Object', 'Error']

#Dataframe to store the error coordinates and error types (only used internally, do not change)
image_error_df = pd.DataFrame(columns=['error_type', 'upper_left_corner', 'lower_right_corner'])

#Function to load the dataset
def load_image_paths_labels(data_directory):

    image_list = glob.glob(data_directory + '**/**/**/**/*.png')

    dataframe = pd.DataFrame(columns=['Image', 'Label'])

    for one_img in image_list:

        one_img = one_img.replace(dataset_directory, '')

        dataframe = dataframe.append({"Image": one_img}, ignore_index=True)

    #Add Status column to allow aborting and continuing the labeling process. 0 = not started processing, 1 = finished processing
    dataframe['Status'] = 0

    dataframe.to_csv(data_directory + 'dataset_images.csv', sep=';', index=False, encoding='utf-8-sig')


#This function will process the finished dataset. Required if additional images were added to the dataset.
#The final dataset will be copied to dataset_directory\Final\
def _process_df(process_folder, process_file):
    shutil.copytree(dataset_patches + process_folder, dataset_directory + 'Final\\' + process_folder)
    shutil.copy2(dataset_patches + process_file, dataset_directory + 'Final\\' + process_file)
    
    patches_df = pd.read_csv(dataset_directory + 'Final\\' + process_file, sep=';')
    patches_df['Label'] = patches_df['Label'] - 1

    patches_df.to_csv(dataset_directory + 'Final\\' + process_file, sep=';', index=False, encoding='utf-8-sig')

    print('Processing completed.\nDataset is ready to be used for training/testing.\nCheck location:\n')
    print(dataset_directory + 'Final\\')

    return

while True:
    query = input('-----------------------------\nDatasets:\n\n(1) Training\n(2) Validation\n(3) Test\n\nor options:\n(4) Process dataset for training\n(5) Tool Info\nPlease select an option: ')
    try:
        query = int(query)
        if query == 1:
           dataset_folder = 'Training_Data\\'
           dataset_file = 'Training_dataset_patches.csv'
           patch_overlap = int(step_size * patch_size)
           break
        
        elif query == 2:
           dataset_folder = 'Validation_Data\\'
           dataset_file = 'Validation_dataset_patches.csv'
           patch_overlap = int(patch_size)
           break

        elif query == 3:
           dataset_folder = 'Test_Data\\'
           dataset_file = 'Test_dataset_patches.csv'
           patch_overlap = int(patch_size)
           break

        elif query == 4:

           while True:
                query2 = input('-----------------------------\nThis will create a copy of the dataset and change the label values from 1-3 to 0-2 for training/testing the CNN.\nDatasets:\n\n(1) Training\n(2) Validation\n(3) Test\n(4) Cancel\nPlease select an option: ')
                try:
                    query2 = int(query2)
                    if query2 == 1:
                       process_folder = 'Training_Data\\'
                       process_file = 'Training_dataset_patches.csv'
                       _process_df(process_folder, process_file)
                       break
        
                    elif query2 == 2:
                       process_folder = 'Validation_Data\\'
                       process_file = 'Validation_dataset_patches.csv'
                       _process_df(process_folder, process_file)
                       break

                    elif query2 == 3:
                       process_folder = 'Test_Data\\'
                       process_file = 'Test_dataset_patches.csv'
                       _process_df(process_folder, process_file)
                       break

                    elif query2 == 4:
                       print('Preprocessing cancelled. Please select a dataset to work on.\n\nDatasets:\n\n(1) Training\n(2) Validation\n(3) Test\n\n')
                       break

                    else:
                       print('Please only enter numbers from 1 - 4.')

                    
                except ValueError:                     
                    print('Please only enter integer numbers.')
                    continue

           

        elif query == 5:
           print('-Click on Next without selecting an error in an image -> Image has no errors. No patches will be created and image will be marked as \'done\' and not shown on the next launch of the tool.\n' +
                 '-Skip Button skips the current image (will be shown again on the next launch of the tool).\n' +
                 '-Select an area of interest and click on the matching button (label). This step can be repeated multiple times for different areas of interest. The image patch will be shown again to correct the label if necessary. All patches which are not in the selected region will be ignored.\n' +
                 '-Select first a Dataset to work on. After completion, start this tool again and select 4 Preprocessing and select the same dataset again to make it usable for training. NOTE: This step is a permanent operation.\n')
        else:
           print('Please only enter numbers from 1 - 5.')
                  
    except ValueError:                     
        print('Please only enter integer numbers.')
        continue


#Counter to store the patch and total number of patches per image. Necessary for saving.
img_filename_counter = 0
#Number of total patches created from a single image
img_filename_max_patches = int(((source_image_dim - (0 if (patch_overlap == patch_size) else patch_overlap))/patch_overlap)
                           * ((source_image_dim - (0 if (patch_overlap == patch_size) else patch_overlap))/patch_overlap))

#Create target directory to store the processed patches
if (os.path.exists(dataset_patches) == False):
    os.makedirs(dataset_patches)

if (os.path.exists(dataset_directory + dataset_folder + 'dataset_images.csv') == False):
    load_image_paths_labels(dataset_directory + dataset_folder)
    dataset_df = pd.read_csv(dataset_directory + dataset_folder + 'dataset_images.csv', sep=';')

else:
    dataset_df = pd.read_csv(dataset_directory + dataset_folder + 'dataset_images.csv', sep=';')


#Check if a csv file exists which contains a list of images to preprocess and their current status.
#Target dataset patches
if (os.path.exists(dataset_patches + dataset_file) == False):
    dataset_patches_df = pd.DataFrame(columns=['Image', 'Label'])

else:
    dataset_patches_df = pd.read_csv(dataset_patches + dataset_file, sep=';')


#Change column values of Status and Uncertain to int to avoid some problems
dataset_df['Status'] = dataset_df['Status'].astype(int)
#dataset_df['Uncertain'] = dataset_df['Status'].astype(int)

#Get the index of the source images, which have to be processed (Status == 0)
unprocessed_images_idx = dataset_df.index[dataset_df['Status'] == False].tolist()

#Total number of source images, which containing an error (len(unprocessed_images_idx))
image_counter = 0
image_counter_max = len(unprocessed_images_idx)


#######################################################################################################
# TF image loading and saving functions begin here
#######################################################################################################
def load_image(img_path):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3, dtype=np.uint8)

    return image

#Function to save a single image patch and write a line in the .csv file containing the image path and the label
def _save_image_patch(img_patch_data, img_patch_label):
    global img_filename_counter
    global dataset_patches_df
        
    img_save = tf.image.encode_png(img_patch_data)
    
    file_name = target_image_path.replace('.png', '_' + str(img_filename_counter) + '.png')

    #Difference from Tool V1:
    #Marked area of an image -> Label 1-3, everything else: Label 0 (=No Error).
    #This tool V2: Marked area of an image -> Label 1-3, everything else will be ignored.
    if img_patch_label != 0:
       tf.io.write_file(file_name, img_save)

       row = pd.Series({'Image': file_name.replace(dataset_patches, ''), 
                     'Label': img_patch_label})

       dataset_patches_df = dataset_patches_df.append(row, ignore_index=True)
       dataset_patches_df.to_csv(dataset_patches + dataset_file, sep=';', index=False, encoding='utf-8-sig')

    #Check if number of saved patches equals max total number of patches per image (starting at 0 -> -1)
    if img_filename_counter == (img_filename_max_patches - 1):
       img_filename_counter = 0

    else:
       img_filename_counter = img_filename_counter + 1


#This function splits the image into blocks which will then be labeled.
def _split_image(window_size, window_overlap, source_dim, image_error_df):

    patch_data, patch_labels = [], []

    error_count = len(image_error_df)

    #Check if there is a 50% overlap between patches or no overlap at all.
    if window_overlap == window_size:
       last_step = 0

    else:
       last_step = window_overlap


    for ypos in range(0, (source_dim - last_step), window_overlap):
            for xpos in range(0, (source_dim - last_step), window_overlap):
                #sliding_window = source_image[ypos : (ypos + window_size), xpos : (xpos + window_size)]
                sliding_window = np.zeros(shape=(window_size, window_size, 3),dtype=np.uint8)
                image_patch = np.zeros(shape=(window_size, window_size, 3),dtype=np.uint8)

                for channel in range(0, 3):
                    sliding_window = source_image[ypos : (ypos + window_size), xpos : (xpos + window_size), channel]
                    image_patch[:, :, channel] = sliding_window


                for err_num in range(0, error_count):
                        
                    error_location = [image_error_df.loc[err_num, 'upper_left_corner'], image_error_df.loc[err_num, 'lower_right_corner']]

                    #Check if the sliding window has an intersection with the error region
                    if (((xpos <= error_location[1][0] <= (xpos + window_size)) and (ypos <= error_location[1][1] <= (ypos + window_size))) 
                        or ((xpos <= error_location[0][0] <= (xpos + window_size)) and (ypos <= error_location[1][1] <= (ypos + window_size)))
                        or ((xpos <= error_location[1][0] <= (xpos + window_size)) and (ypos <= error_location[0][1] <= (ypos + window_size)))
                        or ((xpos <= error_location[0][0] <= (xpos + window_size)) and (ypos <= error_location[0][1] <= (ypos + window_size)))
                        or ((xpos <= error_location[1][0] <= (xpos + window_size)) and ((error_location[0][1] <= ypos) and (error_location[1][1] >= (ypos + window_size))))
                        or ((xpos <= error_location[0][0] <= (xpos + window_size)) and ((error_location[0][1] <= ypos) and (error_location[1][1] >= (ypos + window_size))))
                        or (((error_location[0][0] <= xpos) and (error_location[1][0] >= (xpos + window_size))) and (ypos <= error_location[0][1] <= (ypos + window_size)))
                        or (((error_location[0][0] <= xpos) and (error_location[1][0] >= (xpos + window_size))) and (ypos <= error_location[1][1] <= (ypos + window_size)))
                        or (((error_location[0][0] <= xpos) and (error_location[1][0] >= (xpos + window_size))) and ((error_location[0][1] <= ypos) and (error_location[1][1] >= (ypos + window_size))))
                        or (((xpos <= error_location[0][0]) and ((xpos + window_size)) >= error_location[1][0]) and ((ypos <= error_location[0][1]) and ((ypos + window_size) >= error_location[1][1])))):

                          label = int(image_error_df.loc[err_num, 'error_type'])
                          break


                    #...otherwise, set label to zero because no overlap
                    else:
                        label = 0

                if error_count == 0:
                   label = 0

                patch_data.append(image_patch)
                patch_labels.append(label)

    return patch_data, patch_labels


#Set the label of the patch with a button press and load the next patch
def set_patch_label(event, new_label=0):
    global labels_raw

    if new_label != 4:
       labels_raw[patch_label_raw_idx[patch_n]] = new_label

    _next_patch()

#Load the next patch of the current image or (if it is the last patch) 
#save all patches to disk and load the next image and show the Error selection page.
def _next_patch():
    global patch_n, ax

    if patch_n == (patch_n_max - 1):
        #Enable buttons of the error selection page
        ax_pulverfehler.set_visible(False)
        ax_deltafehler.set_visible(False)
        ax_maschinenfehler.set_visible(False)
        ax_next.set_visible(False)
        ax_skip.set_visible(False)
        ax_quit.set_visible(False)

        #Disable Buttons of the relabel page
        ax_pulverfehler_p.set_visible(False)
        ax_deltafehler_p.set_visible(False)
        ax_maschinenfehler_p.set_visible(False)
        ax_no_error_p.set_visible(False)

        ax.title.set_text('Saving... Please wait.')

        fig.canvas.draw()

        for io in range(img_filename_max_patches):
            _save_image_patch(patches_raw[io], labels_raw[io])

        _load_next_patch()

    
    else:
        patch_n = patch_n + 1

        ax.title.set_text('Patch Label: ' + labels_list[labels_raw[patch_label_raw_idx[patch_n]]])
        img_data_object.set_data(patches_raw[patch_label_raw_idx[patch_n]])

        fig.canvas.draw()


def _load_next_patch():
    global image_error_df, dataset_df, image_counter, source_image_path, source_image_label, target_image_path, source_image, ax, img_data_object

    #Enable buttons of the error selection page
    ax_pulverfehler.set_visible(True)
    ax_deltafehler.set_visible(True)
    ax_maschinenfehler.set_visible(True)
    ax_next.set_visible(True)
    ax_skip.set_visible(True)
    ax_quit.set_visible(True)

    #Disable Buttons of the relabel single patches page
    ax_pulverfehler_p.set_visible(False)
    ax_deltafehler_p.set_visible(False)
    ax_maschinenfehler_p.set_visible(False)
    ax_no_error_p.set_visible(False)

    #Change status of image to '1' (=Done)
    dataset_df.loc[unprocessed_images_idx[image_counter], 'Status'] = 1

    #Save dataset changes
    dataset_df.to_csv(dataset_directory + dataset_folder + 'dataset_images.csv', sep=';', index=False, encoding='utf-8-sig')

    #Increase image counter and load the next image
    image_counter = image_counter + 1

    source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']

    target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

    source_image = load_image(source_image_path)

    ax.title.set_text(source_image_path.replace(dataset_directory, '').split('\\')[-2] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1])
    img_data_object.set_data(source_image)

    fig.canvas.draw()

    #Change Console title as a progressbar
    os.system('title ' + 'Progress... ' + str(round((image_counter/image_counter_max)*100, 2)) + '%')

    #Reset Error Dataframe
    image_error_df = pd.DataFrame(columns=['Image', 'Label'])
    try:
        global upper_left, lower_right
        del upper_left, lower_right
    except:
        pass


#######################################################################################################
# Button controls and functions begin here
#######################################################################################################
#Function to get the coordinates of the selected area in the image
def rect_select(eclick, erelease):
    global upper_left, lower_right

    upper_left = (int(eclick.xdata), int(eclick.ydata))
    lower_right = (int(erelease.xdata), int(erelease.ydata))


#Quit Button
def quit(event):
    plt.close()

#Select a area in the image first. Then click on the Button label.
#Store the coordinates and the label
def save_coordinates(event, patch_label=0):
    global image_error_df

    try:
        new_row = pd.Series({'error_type': patch_label, 
                             'upper_left_corner': upper_left, 
                             'lower_right_corner': lower_right})

        image_error_df = image_error_df.append(new_row, ignore_index=True)

    except:
        print('Please select a region first.')
        ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)


#This function is executed on the button press "Next"
#It will split the image into patches and display the patches which overlap with the selected area again to label them.
#Note: The second labeling is necessary to avoid false labeled patches.
def next(event):
    global image_counter
    global image_error_df, dataset_df
    global source_image_path, source_image_label, target_image_path, source_image
    global patches_raw, labels_raw, patch_label_raw_idx, patch_n_max, patch_n

    #Split the image into patches
    patches_raw, labels_raw = _split_image(patch_size, patch_overlap, source_image_dim, image_error_df)

    #Get indices of patches containing an error and the total number of patches containing an error
    patch_label_raw_idx = list(np.where(labels_raw)[0])
    patch_n_max = len(patch_label_raw_idx)

    #Variable to store the current patch number, initialize with 0
    patch_n = 0

    #Check if any error was selected. if not, load the next image.
    if patch_n_max > 0:
        #Disable buttons of the error selection page
        ax_pulverfehler.set_visible(False)
        ax_deltafehler.set_visible(False)
        ax_maschinenfehler.set_visible(False)
        ax_next.set_visible(False)
        ax_skip.set_visible(False)
        ax_quit.set_visible(False)

        #Enable Buttons of the relabel page
        ax_pulverfehler_p.set_visible(True)
        ax_deltafehler_p.set_visible(True)
        ax_maschinenfehler_p.set_visible(True)
        ax_no_error_p.set_visible(True)

        #update the displayed image patch
        img_data_object.set_data(patches_raw[patch_label_raw_idx[patch_n]])

        fig.canvas.draw()
    
    else:
        
        ax.title.set_text('Saving... Please wait.')

        fig.canvas.draw()

        for io in range(img_filename_max_patches):
            _save_image_patch(patches_raw[io], labels_raw[io])

        #Change status of image to '1' (=Done)
        dataset_df.loc[unprocessed_images_idx[image_counter], 'Status'] = 1

        #Save dataset changes
        dataset_df.to_csv(dataset_directory + dataset_folder + 'dataset_images.csv', sep=';', index=False, encoding='utf-8-sig')

        #Increase image counter and load the next image
        image_counter = image_counter + 1
        
        source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']

        target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

        source_image = load_image(source_image_path)

        ax.title.set_text(source_image_path.replace(dataset_directory, '').split('\\')[-2] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1])
        img_data_object.set_data(source_image)

        fig.canvas.draw()

        #Change Console title as a progressbar
        os.system('title ' + 'Progress... ' + str(round((image_counter/image_counter_max)*100, 2)) + '%')


#This function will skip the current image and load the next one.
#Note: a skipped image will be shown again on the next start of the tool.
def skip(event):
    global image_counter, image_error_df
    global source_image_path, source_image_label, source_image, target_image_path

    if image_counter < (len(unprocessed_images_idx) - 1):
        #Increase image counter and load the next image
        image_counter = image_counter + 1

        source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']

        target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

        source_image = load_image(source_image_path)

        ax.title.set_text(source_image_path.replace(dataset_directory, '').split('\\')[-2] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1])
        img_data_object.set_data(source_image)
    
        fig.canvas.draw()

        #Change Console title as a progressbar
        os.system('title ' + 'Progress... ' + str(round((image_counter/image_counter_max)*100, 2)) + '%')

        #Reset Error Dataframe
        image_error_df = pd.DataFrame(columns=['Image', 'Label'])
        
        try:
            global upper_left, lower_right
            del upper_left, lower_right
        except:
            pass

    else:
        print('No further images to label')
        ctypes.windll.user32.FlashWindow(ctypes.windll.kernel32.GetConsoleWindow(), True)


#######################################################################################################
# GUI setup
#######################################################################################################
print(image_counter)
print(unprocessed_images_idx)
source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']

target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

source_image = load_image(source_image_path)


fig, ax = plt.subplots()
plt.axis('off')
ax.title.set_text(source_image_path.replace(dataset_directory, '').split('\\')[-2] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1])

img_data_object = ax.imshow(source_image, interpolation='nearest')

#Button Powder
ax_pulverfehler = plt.axes([0.08, 0.06, 0.2, 0.05])
b_pulverfehler = Button(ax_pulverfehler, 'Powder (good)')
#Button Object
ax_deltafehler = plt.axes([0.30, 0.06, 0.2, 0.05])
b_deltafehler = Button(ax_deltafehler, 'Object')
#Button Error
ax_maschinenfehler = plt.axes([0.52, 0.06, 0.2, 0.05])
b_maschinenfehler = Button(ax_maschinenfehler, 'Error')
#Button Next
ax_next = plt.axes([0.74, 0.06, 0.2, 0.05])
b_next = Button(ax_next, 'Next')
#Button Skip
ax_skip = plt.axes([0.74, 0.005, 0.2, 0.05])
b_skip = Button(ax_skip, 'Skip')
#Button Quit
ax_quit = plt.axes([0.08, 0.005, 0.15, 0.05])
b_quit = Button(ax_quit, 'Quit')

b_pulverfehler.on_clicked(lambda x: save_coordinates(x, 1))
b_deltafehler.on_clicked(lambda x: save_coordinates(x, 2))
b_maschinenfehler.on_clicked(lambda x: save_coordinates(x, 3))
b_next.on_clicked(next)
b_skip.on_clicked(skip)
b_quit.on_clicked(quit)


#Buttons of the second Layer

#Button Powder (good)
ax_pulverfehler_p = plt.axes([0.0801, 0.06, 0.2, 0.05])
b_pulverfehler_p = Button(ax_pulverfehler_p, 'Powder (good)')
#Button Object
ax_deltafehler_p = plt.axes([0.3101, 0.06, 0.2, 0.05])
b_deltafehler_p = Button(ax_deltafehler_p, 'Object')
#Button Error
ax_maschinenfehler_p = plt.axes([0.5201, 0.06, 0.2, 0.05])
b_maschinenfehler_p = Button(ax_maschinenfehler_p, 'Error')
#Button Ignore Patch
ax_no_error_p = plt.axes([0.7401, 0.06, 0.2, 0.05])
b_no_error_p = Button(ax_no_error_p, 'Ignore Patch')

b_no_error_p.on_clicked(lambda y: set_patch_label(y, 0))
b_pulverfehler_p.on_clicked(lambda y: set_patch_label(y, 1))
b_deltafehler_p.on_clicked(lambda y: set_patch_label(y, 2))
b_maschinenfehler_p.on_clicked(lambda y: set_patch_label(y, 3))

ax_pulverfehler_p.set_visible(False)
ax_deltafehler_p.set_visible(False)
ax_maschinenfehler_p.set_visible(False)
ax_no_error_p.set_visible(False)

fig.canvas.draw()

_ = RectangleSelector(ax, rect_select,
                      drawtype='box', useblit=True,
                      button=[1], 
                      minspanx=5, minspany=5,
                      spancoords='pixels',
                      interactive=True)


plt.show()
