There are two versions of this Label Tool.

V1 (left) and V2 (right):

<table>
  <tr>
    <td>Tool V1</td>
    <td>Tool V2</td>
  </tr>
  <tr>
    <td><img src="/Label_Tool/tool_v1.png" width=300></td>
    <td><img src="/Label_Tool/tool_v2.png" width=300></td>
  </tr>
 </table>
 
Tool V2 was used in this work and is based on V1 with a few changes.

Main differences between the Versions:

V1:

The focus on this tool V1 was to label only the occurring errors during the printing process and their error type class.

1. Select a Dataset to process.
2. Select a Region of Interest (ROI) in the displayed image and the Error Type Button and click on Next afterwards.
3. The Tool will now display all patches within the selected ROI to confirm or, if required, to adjust the label for the displayed patch.
4. Once all patch labels have been examined, the tool will save the patches to disk. The label for each patch will be stored in an .csv file. All patches outside of the selected ROI will be also stored to disk. With the difference, that their label will be set to "No Error".

Note:
-Skipping a slice image with the skip button will not mark the layer image as "completed". Therefore, the tool will on the next start continue with this image, as its label is still 0 (=not examined. 1=examination/layer image is complete).

V2:

In contrast to V1, this tool V2 will save only the selected areas to disk and discard every patch, which is not inside of the selected Region of Interest. The label classes have also been changed. Instead of "No Error" and the different types of error classes, this tool only has the classes: "Powder (good)", "Object" and "Error" (any type of visible error falls in this class). Because this tool is based on V1, an additional step is required, before the dataset can be used for training/evaluation. Once the examination of all layer/patch images is completed, the tool has to be launched again and the option "(5)" must be selected. Inside, the finished dataset must be selected for pre-processing. This step subtracts from each patch label "1", so that the final dataset has labels going from 0-2 instead of 1-3.

1. Select a Dataset to process.
2. Select a Region of Interest (ROI) in the displayed image and the Error Type Button and click on Next afterwards.
3. The Tool will now display all patches within the selected ROI to confirm or, if required, to adjust the label for the displayed patch.
4. Once all patch labels have been examined, the tool will save the patches to disk. The label for each patch will be stored in an .csv file. All patches outside of the selected ROI will be discarded.
5. The tool must be launched once again after completion of the dataset labelling, to normalize the labels. This step can be executed by selection the pre-processing option (5) upon launch.
