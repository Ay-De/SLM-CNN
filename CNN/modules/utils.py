import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns

def list_shuffle(*arrays, **seeds):
    """
    This function shuffles the provided lists row wise, while keeping the correlation inside the list rows.
    For reasons of reproducibility, shuffling happens with given seeds.
    """

    num_arrays = len(arrays)

    if num_arrays == 0:
       raise ValueError('At least one array is required.')

    num_shuffles = len(seeds)
    
    if num_shuffles == 0:
       raise ValueError('At least one seed is required.')


    if (num_arrays > 1):
       if not all(len(x) == len(arrays[0]) for x in arrays):
          raise ValueError('All lists need to have the same length.')

       array = list(zip(*arrays))
       zipped = True

    else:
       array, = arrays
       zipped = False

    for s in seeds:
        random.seed(seeds.get(s))
        random.shuffle(array)

    if zipped == True:
       *arrays, = zip(*array)
       for i in range(0, num_arrays):
           arrays[i] = list(arrays[i])

       return arrays

    else:
       return array



def plot_loss_and_accuracy(acc, val_acc, loss, val_loss, epochs):
    """
    Function to plot the accuracy and loss over the trained epochs
    
    Code source:
    https://www.tensorflow.org/tutorials/images/classification#visualize_training_results
    """

    epochs_range = range(0, epochs, 1)


    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validierung', linestyle='--', linewidth=2)
    #Note: Baseline was calculated by the sklearn DummyClassifier
    plt.hlines(0.46, 0, epochs, linestyles=':', label='Baseline', color='g')
    plt.xlabel('Epochen', fontweight='bold', labelpad=10)
    plt.xticks(range(0, epochs, 10))
    plt.ylabel('Korrektklassifizierungsrate', fontweight='bold', labelpad=10)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validierung', linestyle='--', linewidth=2)
    plt.xlabel('Epochen', fontweight='bold', labelpad=10)
    plt.xticks(range(0, epochs, 10))
    plt.ylabel('Straffunktion', fontweight='bold', labelpad=10)
    plt.ylim([0, plt.ylim()[1]])
    plt.legend(loc='upper right')
    plt.tight_layout()


def plot_cm(labels, predictions, multiclass_classes):
    '''
    Plot Confusion Matrix
    '''
    cm = confusion_matrix(labels, predictions, normalize='true')
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='.2%', xticklabels=multiclass_classes, yticklabels=multiclass_classes, cmap='Blues', cbar=False, square=True)

    plt.yticks(rotation=0)
    plt.ylabel('Tatsächliche Klasse', fontweight='bold', labelpad=10)
    plt.xlabel('Vorhergesagte Klasse', fontweight='bold', labelpad=20)
    plt.tight_layout()


#Source for the Integrated Gradients implementation:
#https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
#With some modifications to make it work on this dataset
#This functions creates the linear interpolated images from a given single image
def interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images

def compute_gradients(model, images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

@tf.function
def integrated_gradients(model, baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.    
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(model=model, images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)    

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


def plot_img_attributions(model, baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          cmap=None,
                          overlay_alpha=0.4):

    #save the original image to show it later
    original_image = image
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    attributions = integrated_gradients(model=model, 
                                      baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)

    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_title('Baseline image')
    axs[0, 0].imshow(baseline)
    axs[0, 0].axis('off')

    axs[0, 1].set_title('Original image')
    axs[0, 1].imshow(original_image)
    axs[0, 1].axis('off')

    axs[1, 0].set_title('Attribution mask')
    axs[1, 0].imshow(attribution_mask, cmap=cmap)
    axs[1, 0].axis('off')

    axs[1, 1].set_title('Overlay')
    axs[1, 1].imshow(attribution_mask, cmap=cmap)
    axs[1, 1].imshow(image, alpha=overlay_alpha)
    axs[1, 1].axis('off')

    plt.tight_layout()

    return fig