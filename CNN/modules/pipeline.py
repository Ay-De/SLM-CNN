import tensorflow as tf

#Training: Load image, (semi randomly) augmentate the image and subtract per image mean and std
def _load_and_preprocess_image_train(image_path, label=None):

    image = image_path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    image = tf.image.per_image_standardization(image)
    
    if label == None:
       return image

    else:
        return image, label

#Validation/Test: Load image and subtract per image mean and std
def _load_and_preprocess_image_test(image_path, label=None):

    image = image_path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    
    if label == None:
       return image

    else:
        return image, label



#This function generates the image batches
#if shuffle is True: dataset will be shuffled before batching
#if predict is True: dataset will not be repeated (important for the last batch in case of predicting the validation/test data)
def batch_dataset(dataset, batch_size, shuffle=True, predict=False):

    if shuffle == True:
       dataset = dataset.shuffle(buffer_size=20 * 1000 * batch_size)
    
    #For training: augmentate the images (random flip after loading)
    if predict == False:
       dataset = dataset.repeat()
       dataset = dataset.map(_load_and_preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #For validation/test: No image augmentation upon creating the batch
    else:
       dataset = dataset.repeat(1)
       dataset = dataset.map(_load_and_preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #Create the batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset
