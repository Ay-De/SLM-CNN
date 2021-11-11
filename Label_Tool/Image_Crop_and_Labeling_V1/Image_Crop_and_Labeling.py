import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import glob
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #1: filter out info logs and 2 and 3, 2: filter out warning logs and 3, 3: filter out only error logs
import tensorflow as tf
import ctypes

#######################################################################################################
# Some global variables (i know, big nono)
#######################################################################################################
os.chdir('..')
#Location of the dataset
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

#Label type to use.
multiclass_labels_type = 'Label_Multiclass_V2'

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
#No Error: 0
#Powder Error: 1
#Delta Error: 2
#Machine Error: 3
labels_list = ['No Error', 'Powder Error', 'Delta Error', 'Machine Error']

#Dataframe to store the error coordinates and error types
image_error_df = pd.DataFrame(columns=['error_type', 'upper_left_corner', 'lower_right_corner'])

#Function to load the dataset
def load_image_paths_labels(data_directory):

    complete_good_df = pd.DataFrame(columns=['Image', 'Label_Binary', multiclass_labels_type])
    complete_bad_df = complete_good_df.copy()
    project_list = glob.glob(data_directory + '**/**/**/image_label.csv')

    for project_folder in project_list:
        dataframe = pd.read_csv(project_folder, sep=';')
        #Drop the first few images, because they are showing an incomplete powderbed
        dataframe = dataframe[dataframe['Drop_Image'] != 1]

        #Drop column 'Drop_Images'
        dataframe.drop(columns=['Drop_Image'], axis=1, inplace=True)

        #Get the path to the stored images
        project_folder = project_folder.replace('logs\\image_label.csv', 'logs\\exposures\\')
        project_folder = project_folder.replace(dataset_directory, '')
        #Replace image filename in the dataframe with the path + the image filename
        dataframe['Image'] = project_folder + dataframe['Image'].astype(str)

        #Add Status column to allow aborting and continuing the labeling process. 0 = not started processing, 1 = finished processing
        dataframe['Status'] = 0

        #Add Uncertain column to mark uncertain images. 1 = Image with uncertain errors
        #dataframe['Uncertain'] = 0

        #Seperate the images between the ones showing an error, and the ones showing no error
        bad_images_df = dataframe[dataframe['Label_Binary'] == 1].copy().reset_index(drop=True)
        good_images_df = dataframe[dataframe['Label_Binary'] == 0].copy().reset_index(drop=True)

        complete_bad_df = pd.concat([complete_bad_df, bad_images_df], axis=0)
        complete_good_df = pd.concat([complete_good_df, good_images_df], axis=0)
    
    complete_bad_df.to_csv(data_directory + 'dataset_bad_images.csv', sep=';', index=False, encoding='utf-8-sig')
    complete_good_df.to_csv(data_directory + 'dataset_good_images.csv', sep=';', index=False, encoding='utf-8-sig')



print('Datasets:\n\n(1) Training\n(2) Validation\n(3) Test\n(4) Tool Info\n')
while True:
    query = input('Please select a dataset: ')
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
           print('-Click on Next without selecting an error in an image -> Image has no errors. All patches will have the label no error.\n' +
                 '-Skip Button skips the current image (will be reshown on the next launch of the tool).\n' +
                 '-Select an area of interest and click on the matching button. The image will be shown again to correct the label if necessary. All patches which are not in the selected region will be labeled as "No Error".\n')

        else:
           print('Please only enter numbers from 1 - 4.')
                  
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

if ((os.path.exists(dataset_directory + dataset_folder + 'dataset_bad_images.csv') & 
    os.path.exists(dataset_directory + dataset_folder + 'dataset_good_images.csv')) == False):
    load_image_paths_labels(dataset_directory + dataset_folder)
    dataset_df = pd.read_csv(dataset_directory + dataset_folder + 'dataset_bad_images.csv', sep=';')

else:
    dataset_df = pd.read_csv(dataset_directory + dataset_folder + 'dataset_bad_images.csv', sep=';')

#Target dataset patches
if (os.path.exists(dataset_patches + dataset_file) == False):
    dataset_patches_df = pd.DataFrame(columns=['Image', 'Label_Binary', multiclass_labels_type])

else:
    dataset_patches_df = pd.read_csv(dataset_patches + dataset_file, sep=';')


#Change column values of Status and Uncertain to int to avoid some problems
dataset_df['Status'] = dataset_df['Status'].astype(int)
#dataset_df['Uncertain'] = dataset_df['Status'].astype(int)

#Change the remaining Label_Multiclass column to integer numbers
dataset_df[('Label_Multiclass' if multiclass_labels_type == 'Label_Multiclass_V2' else 'Label_Multiclass_V2')] = dataset_df[('Label_Multiclass' if multiclass_labels_type == 'Label_Multiclass_V2' else 'Label_Multiclass_V2')].astype(int)
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

    tf.io.write_file(file_name, img_save)

    row = pd.Series({'Image': file_name.replace(dataset_patches, ''), 
                     'Label_Binary': (0 if img_patch_label == 0 else 1), 
                     multiclass_labels_type: img_patch_label})

    dataset_patches_df = dataset_patches_df.append(row, ignore_index=True)
    dataset_patches_df.to_csv(dataset_patches + dataset_file, sep=';', index=False, encoding='utf-8-sig')

    #Check if number of saved patches equals max total number of patches per image (starting at 0 -> -1)
    if img_filename_counter == (img_filename_max_patches - 1):
       img_filename_counter = 0

    else:
       img_filename_counter = img_filename_counter + 1

#This function splits the image into blocks which will then be labelled.
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
        ax_ok_p.set_visible(False)

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


#Set current image to done (change statusbit to 1) and load the next image
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
    ax_ok_p.set_visible(False)

    #Change status of image to '1' (=Done)
    dataset_df.loc[unprocessed_images_idx[image_counter], 'Status'] = 1

    #Save dataset changes
    dataset_df.to_csv(dataset_directory + dataset_folder + 'dataset_bad_images.csv', sep=';', index=False, encoding='utf-8-sig')

    #Increase image counter and load the next image
    image_counter = image_counter + 1

    source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']
    source_image_label = dataset_df.loc[unprocessed_images_idx[image_counter], multiclass_labels_type]

    target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

    source_image = load_image(source_image_path)

    ax.title.set_text('Source Label: ' + r'$\bf{' + labels_list[source_image_label] + '}$' + ' ' + '(' + source_image_path.replace(dataset_directory, '').split('\\')[-4] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1] + ')')
    img_data_object.set_data(source_image)

    fig.canvas.draw()

    #Change Console title as a progressbar
    os.system('title ' + 'Progress... ' + str(round((image_counter/image_counter_max)*100, 2)) + '%')

    #Reset Error Dataframe
    image_error_df = pd.DataFrame(columns=['Image', 'Label_Binary', multiclass_labels_type])
    try:
        global upper_left, lower_right
        del upper_left, lower_right
    except:
        pass


#######################################################################################################
# Button controls and functions begin here
#######################################################################################################
def rect_select(eclick, erelease):
    global upper_left, lower_right

    upper_left = (int(eclick.xdata), int(eclick.ydata))
    lower_right = (int(erelease.xdata), int(erelease.ydata))



def quit(event):
    plt.close()


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
        ax_ok_p.set_visible(True)

        ax.title.set_text('Patch Label: ' + labels_list[labels_raw[patch_label_raw_idx[patch_n]]])
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
        dataset_df.to_csv(dataset_directory + dataset_folder + 'dataset_bad_images.csv', sep=';', index=False, encoding='utf-8-sig')

        #Increase image counter and load the next image
        image_counter = image_counter + 1

        source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']
        source_image_label = dataset_df.loc[unprocessed_images_idx[image_counter], multiclass_labels_type]

        target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

        source_image = load_image(source_image_path)

        ax.title.set_text('Source Label: ' + r'$\bf{' + labels_list[source_image_label] + '}$' + ' ' + '(' + source_image_path.replace(dataset_directory, '').split('\\')[-4] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1] + ')')
        img_data_object.set_data(source_image)

        fig.canvas.draw()

        #Change Console title as a progressbar
        os.system('title ' + 'Progress... ' + str(round((image_counter/image_counter_max)*100, 2)) + '%')


    
def skip(event):
    global image_counter, image_error_df
    global source_image_path, source_image_label, source_image, target_image_path

    if image_counter < len(unprocessed_images_idx):
        #Increase image counter and load the next image
        image_counter = image_counter + 1

        source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']
        source_image_label = dataset_df.loc[unprocessed_images_idx[image_counter], multiclass_labels_type]

        target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

        source_image = load_image(source_image_path)

        ax.title.set_text('Source Label: ' + r'$\bf{' + labels_list[source_image_label] + '}$' + ' ' + '(' + source_image_path.replace(dataset_directory, '').split('\\')[-4] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1] + ')')
        img_data_object.set_data(source_image)
    
        fig.canvas.draw()

        #Change Console title as a progressbar
        os.system('title ' + 'Progress... ' + str(round((image_counter/image_counter_max)*100, 2)) + '%')

        #Reset Error Dataframe
        image_error_df = pd.DataFrame(columns=['Image', 'Label_Binary', multiclass_labels_type])
        
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
source_image_path = dataset_directory + dataset_df.loc[unprocessed_images_idx[image_counter], 'Image']
source_image_label = dataset_df.loc[unprocessed_images_idx[image_counter], multiclass_labels_type]

target_image_path = source_image_path.replace(dataset_directory, dataset_patches)

source_image = load_image(source_image_path)


fig, ax = plt.subplots()
plt.axis('off')
ax.title.set_text('Source Label: ' + r'$\bf{' + labels_list[source_image_label] + '}$' + ' ' + '(' + source_image_path.replace(dataset_directory, '').split('\\')[-4] + '\\' + source_image_path.replace(dataset_directory, '').split('\\')[-1] + ')')

img_data_object = ax.imshow(source_image, interpolation='nearest')

#Button Powder Error
ax_pulverfehler = plt.axes([0.08, 0.06, 0.2, 0.05])
b_pulverfehler = Button(ax_pulverfehler, 'Powder Error')
#Button Delta Error
ax_deltafehler = plt.axes([0.30, 0.06, 0.2, 0.05])
b_deltafehler = Button(ax_deltafehler, 'Delta Error')
#Button Machine Error
ax_maschinenfehler = plt.axes([0.52, 0.06, 0.2, 0.05])
b_maschinenfehler = Button(ax_maschinenfehler, 'Machine Error')
#Button Next
ax_next = plt.axes([0.74, 0.06, 0.2, 0.05])
b_next = Button(ax_next, 'Next')
#Button Quit
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

#Button Powder Error
ax_pulverfehler_p = plt.axes([0.0801, 0.06, 0.2, 0.05])
b_pulverfehler_p = Button(ax_pulverfehler_p, 'Powder Error')
#Button Delta Error
ax_deltafehler_p = plt.axes([0.3101, 0.06, 0.2, 0.05])
b_deltafehler_p = Button(ax_deltafehler_p, 'Delta Error')
#Button Machine Error
ax_maschinenfehler_p = plt.axes([0.5201, 0.06, 0.2, 0.05])
b_maschinenfehler_p = Button(ax_maschinenfehler_p, 'Machine Error')
#Button No Error
ax_no_error_p = plt.axes([0.7401, 0.06, 0.2, 0.05])
b_no_error_p = Button(ax_no_error_p, 'No Error')
#Button Next
ax_ok_p = plt.axes([0.31, 0.005, 0.41, 0.05])
b_ok_p = Button(ax_ok_p, 'Label Ok')

b_no_error_p.on_clicked(lambda y: set_patch_label(y, 0))
b_pulverfehler_p.on_clicked(lambda y: set_patch_label(y, 1))
b_deltafehler_p.on_clicked(lambda y: set_patch_label(y, 2))
b_maschinenfehler_p.on_clicked(lambda y: set_patch_label(y, 3))
b_ok_p.on_clicked(lambda y: set_patch_label(y, 4))

ax_pulverfehler_p.set_visible(False)
ax_deltafehler_p.set_visible(False)
ax_maschinenfehler_p.set_visible(False)
ax_no_error_p.set_visible(False)
ax_ok_p.set_visible(False)

fig.canvas.draw()

_ = RectangleSelector(ax, rect_select,
                      drawtype='box', useblit=True,
                      button=[1], 
                      minspanx=5, minspany=5,
                      spancoords='pixels',
                      interactive=True)


plt.show()
