# **SLM-CNN**

This repository contains the source code, trained CNN model and examples for the proceedings paper:

CNN based powder bed monitoring and anomaly detection for the selective laser melting process

## Dataset

## Installation

If you want to run the model on the provided samples, please install the requirements.txt first:
```
pip install -r requirements.txt
```

## Architecture
'''Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
image (InputLayer)           [(None, 128, 128, 3)]     0
_________________________________________________________________
conv2d (Conv2D)              (None, 128, 128, 32)      896
_________________________________________________________________
batch_normalization (BatchNo (None, 128, 128, 32)      128
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 64)        18496
_________________________________________________________________
batch_normalization_1 (Batch (None, 64, 64, 64)        256
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 64)        0
_________________________________________________________________
dropout (Dropout)            (None, 32, 32, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 128)       73856
_________________________________________________________________
batch_normalization_2 (Batch (None, 32, 32, 128)       512
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 16, 128)       0
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 16, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 256)       295168
_________________________________________________________________
batch_normalization_3 (Batch (None, 16, 16, 256)       1024
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 8, 256)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 8, 8, 256)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 256)         590080
_________________________________________________________________
batch_normalization_4 (Batch (None, 8, 8, 256)         1024
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 4, 4, 256)         0
_________________________________________________________________
dropout_3 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0
_________________________________________________________________
dense (Dense)                (None, 512)               2097664
_________________________________________________________________
dropout_4 (Dropout)          (None, 512)               0
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 1539
=================================================================
Total params: 3,080,643
Trainable params: 3,079,171
Non-trainable params: 1,472
_________________________________________________________________
''' <img src="./model_plot.png" width=30% height=30%> 

## Results
The model architecture, as seen in the previous section, was trained five times. Classification results were obtained by averaging the classification results of the test set.

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| Powder | 0.8418±0.0279 | 0.9804±0.0071 | 0.9057±0.0190 |
| Object | 0.9039±0.0150 | 0.7540±0.0291 | 0.8216±0.0127 |
| Error | 0.8343±0.0150 | 0.8607±0.0213 | 0.8471±0.0113 |
| Accuracy | | | 0.8574±0.0080 |
| Macro Average Accuracy | 0.8600±0.0070 | 0.8650±0.0076 | 0.8581±0.0084 |
| Weighted Average Accuracy | 0.8618±0.0064 | 0.8574±0.0080 | 0.8552±0.0082 |

## **Citation**

Please consider citing this paper if you deem it helpful in your research:

```
Placeholder
for
bibtex cite
```

## **Contact**

aydin [.] deliktas [_at ] th-ab [.] de

michael [.] moeckel [_at_] th-ab [.] de
